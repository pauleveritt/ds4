#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL line %d\n", __LINE__);                                           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
/* Regression: resident grouped routed MoE with Q4_K gate/up, a Q6_K down and
 * no SwiGLU clamp must match a CPU FP32 reference when the SwiGLU
 * intermediate exceeds FP16. Before 3893b22 the grouped path staged it in
 * half precision and overflowed. No model required. */
typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12], qs[128];
} q4k;
typedef struct {
    uint8_t ql[128], qh[64];
    int8_t scales[16];
    uint16_t d;
} q6k;
static uint32_t seed = 1;
static uint32_t rnd(void) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}
static float h2f(uint16_t h) {
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}
static void q4k_row(const q4k *b, float *y) {
    const float d = h2f(b->d), dmin = h2f(b->dmin);
    const uint8_t *q = b->qs, *s = b->scales;
    for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32)
        for (int half = 0; half < 2; half++) {
            int k = is + half, sc, m;
            if (k < 4) {
                sc = s[k] & 63;
                m = s[k + 4] & 63;
            } else {
                sc = (s[k + 4] & 0xF) | ((s[k - 4] >> 6) << 4);
                m = (s[k + 4] >> 4) | ((s[k] >> 6) << 4);
            }
            for (int l = 0; l < 32; l++)
                *y++ = d * sc * (half ? q[l] >> 4 : q[l] & 0xF) - dmin * m;
        }
}
static void q6k_row(const q6k *b, float *y) {
    const float d = h2f(b->d);
    const uint8_t *ql = b->ql, *qh = b->qh;
    const int8_t *sc = b->scales;
    for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            y[l] = d * sc[is] * (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = d * sc[is + 2] * (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = d * sc[is + 4] * (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = d * sc[is + 6] * (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
}
int main(void) {
    enum { D = 256, H = 256, E = 32, N = 8, T = 257 };
    uint64_t row = sizeof(q4k), expert = H * row, tensor = E * expert, downrow = sizeof(q6k),
             downexpert = D * downrow, bytes = 2 * tensor + E * downexpert;
    CHECK(row == 144 && downrow == 210);
    FILE *file = tmpfile();
    CHECK(file);
    CHECK(!ftruncate(fileno(file), bytes));
    q4k *model = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(file), 0);
    CHECK(model != MAP_FAILED);
    for (size_t i = 0; i < 2 * tensor / sizeof(q4k); i++) {
        model[i].d = 0x1800;
        model[i].dmin = 0x1400;
        for (int j = 0; j < 12; j++)
            model[i].scales[j] = rnd();
        for (int j = 0; j < 128; j++)
            model[i].qs[j] = rnd();
    }
    q6k *down = (q6k *)((char *)model + 2 * tensor);
    for (size_t i = 0; i < E * downexpert / sizeof(q6k); i++) {
        for (int j = 0; j < 128; j++)
            down[i].ql[j] = rnd();
        for (int j = 0; j < 64; j++)
            down[i].qh[j] = rnd();
        for (int j = 0; j < 16; j++)
            down[i].scales[j] = (rnd() % 15) - 7;
        down[i].d = 0x1800;
    }
    CHECK(!msync(model, bytes, MS_SYNC));
    /* Dequantized weights: gate and up [E][H][D], down [E][D][H]. */
    float *gw = malloc((size_t)E * H * D * 4), *uw = malloc((size_t)E * H * D * 4),
          *dw = malloc((size_t)E * D * H * 4);
    CHECK(gw && uw && dw);
    for (int e = 0; e < E; e++)
        for (int r = 0; r < H; r++) {
            q4k_row(&model[(size_t)e * H + r], gw + ((size_t)e * H + r) * D);
            q4k_row(&model[(size_t)(E + e) * H + r], uw + ((size_t)e * H + r) * D);
        }
    for (int e = 0; e < E; e++)
        for (int r = 0; r < D; r++)
            q6k_row(&down[(size_t)e * D + r], dw + ((size_t)e * D + r) * H);

    CHECK(ds4_gpu_init());
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    CHECK(ds4_gpu_set_model_map(model, bytes));
    float *x = malloc(T * D * 4), *w = malloc(T * N * 4), *got = malloc(T * D * 4),
          *ref = malloc(T * D * 4), *mid = malloc(H * 4);
    int32_t *ids = malloc(T * N * 4);
    CHECK(x && w && got && ref && mid && ids);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < N; k++) {
            ids[t * N + k] = (t * 13 + k * 17) % E;
            w[t * N + k] = (k + 1) / 36.f;
        }
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(T * D * 4), *wt = ds4_gpu_tensor_alloc(T * N * 4),
                   *it = ds4_gpu_tensor_alloc(T * N * 4),
                   *midt = ds4_gpu_tensor_alloc(T * N * H * 4),
                   *out = ds4_gpu_tensor_alloc(T * D * 4);
    CHECK(xt && wt && it && midt && out);
    CHECK(ds4_gpu_tensor_write(wt, 0, w, T * N * 4));
    CHECK(ds4_gpu_tensor_write(it, 0, ids, T * N * 4));

    /* "normal" keeps the intermediate inside FP16 and checks the reference;
     * "large" pushes it past FP16's 65504, which is the defect. */
    const struct {
        const char *name;
        float divisor;
    } scales[] = {{"normal", 256.f}, {"large", 0.256f}};
    const int counts[] = {95, 96, 257}; /* 95 stays token-major; 96+ is grouped */
    const float tol = 1e-2f;
    int failed = 0;
    for (unsigned s = 0; s < 2; s++) {
        seed = 7;
        for (int i = 0; i < T * D; i++)
            x[i] = ((int)(rnd() % 101) - 50) / scales[s].divisor;
        CHECK(ds4_gpu_tensor_write(xt, 0, x, T * D * 4));
        float mid_max = 0;
        for (int t = 0; t < T; t++) {
            float *o = ref + (size_t)t * D;
            memset(o, 0, D * 4);
            for (int k = 0; k < N; k++) {
                int e = ids[t * N + k];
                for (int r = 0; r < H; r++) {
                    const float *g = gw + ((size_t)e * H + r) * D, *u = uw + ((size_t)e * H + r) * D;
                    double gs = 0, us = 0;
                    for (int i = 0; i < D; i++) {
                        gs += (double)g[i] * x[t * D + i];
                        us += (double)u[i] * x[t * D + i];
                    }
                    mid[r] = (float)(gs / (1.0 + exp(-gs)) * us);
                    if (fabsf(mid[r]) > mid_max)
                        mid_max = fabsf(mid[r]);
                }
                for (int r = 0; r < D; r++) {
                    const float *dr = dw + ((size_t)e * D + r) * H;
                    double acc = 0;
                    for (int i = 0; i < H; i++)
                        acc += (double)dr[i] * mid[i];
                    o[r] += w[t * N + k] * (float)acc;
                }
            }
        }
        for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            int n = counts[c];
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_glm_routed_moe_batch_tensor(
                out, midt, model, bytes, 0, tensor, 2 * tensor, 12, 12, 14, expert, row, expert,
                row, downexpert, downrow, D, H, D, it, wt, E, N, 0.f, 0, xt, n, N * H, true));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(out, 0, got, n * D * 4));
            float ref_max = 0, err_max = 0;
            int nonfinite = 0;
            for (int i = 0; i < n * D; i++) {
                if (!isfinite(got[i]))
                    nonfinite++;
                else if (fabsf(got[i] - ref[i]) > err_max)
                    err_max = fabsf(got[i] - ref[i]);
                if (fabsf(ref[i]) > ref_max)
                    ref_max = fabsf(ref[i]);
            }
            float rel = ref_max > 0 ? err_max / ref_max : err_max;
            int bad = nonfinite != 0 || !(rel < tol);
            printf("scale=%s mid_max=%g tokens=%d nonfinite=%d ref_max=%g rel_err=%g %s\n",
                   scales[s].name, mid_max, n, nonfinite, ref_max, rel, bad ? "FAIL" : "ok");
            failed |= bad;
        }
    }
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(it);
    ds4_gpu_tensor_free(midt);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(gw);
    free(uw);
    free(dw);
    free(x);
    free(w);
    free(got);
    free(ref);
    free(mid);
    free(ids);
    munmap(model, bytes);
    fclose(file);
    return failed;
}
