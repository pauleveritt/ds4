/* Model-free test for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE)
 * full-attention CPU layer core.
 *
 * The layer is exposed by a test hook that ds4.c compiles under
 * -DDS4_TEST_HOOKS.  The hook is not declared in ds4.h; the contract is
 * duplicated here, exactly as tests/test_qwen35_tokenizer.c duplicates the
 * other qwen35 hooks.  Synthetic weights live in one anonymous mmap and are
 * handed to the hook as explicit f32 pointers, so the test measures the layer
 * and not the quantizer.
 *
 * The reference below is a plain scalar implementation of the same math:
 * pre-attention RMSNorm, q/k/v projections (attn_q is double width and only
 * its query half is used in this increment), GQA grouping (n_head query heads
 * share n_head_kv KV heads by h / (n_head / n_head_kv)), a causal softmax over
 * key_pos <= query_pos, the output projection, the attention residual and the
 * post-attention RMSNorm.
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-attn
 *   ./external/ds4/tests/test_qwen35_attn
 */
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"

/* Ornith-1.5-35B-A3B (DS4_SHAPE_ORNITH15) full-attention geometry. */
enum {
    N_EMBD      = 2048,
    N_HEAD      = 16,
    N_HEAD_KV   = 2,
    HEAD_DIM    = 256,
    N_ROT       = 64,
    Q_DIM       = N_HEAD * HEAD_DIM,      /* 4096 */
    Q_GATE_DIM  = 2 * Q_DIM,              /* 8192 */
    KV_DIM      = N_HEAD_KV * HEAD_DIM,   /* 512 */
    GROUP       = N_HEAD / N_HEAD_KV,     /* 8   */
    N_TOKENS    = 3,
    MODEL_BYTES = 128 * 1024 * 1024,
};

/* Must match the struct ds4.c defines for the hook. */
typedef struct {
    const float *attn_q;       /* [Q_GATE_DIM][N_EMBD] */
    const float *attn_k;       /* [KV_DIM][N_EMBD] */
    const float *attn_v;       /* [KV_DIM][N_EMBD] */
    const float *attn_output;  /* [N_EMBD][Q_DIM] */
    const float *attn_norm;    /* [N_EMBD] pre-attention */
    const float *ffn_norm;     /* [N_EMBD] post-attention */
    const float *attn_q_norm;  /* [HEAD_DIM], Task 2 */
    const float *attn_k_norm;  /* [HEAD_DIM], Task 2 */
    const float *x;            /* [n_tokens][N_EMBD] */
    float       *out;          /* [n_tokens][N_EMBD] */
    uint32_t     n_tokens;
    uint32_t     il;
    uint32_t     n_rot;        /* Task 2; 0 disables */
    uint32_t     pos0;         /* Task 2 */
    float        rope_freq_base;
} ds4_test_qwen35_attn_args;

int ds4_test_qwen35_attn_forward(const ds4_test_qwen35_attn_args *args);

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void require_close(const char *what, float actual, float expected,
                          float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.9g, expected %.9g (tolerance %.9g)\n",
                what, actual, expected, tolerance);
        exit(1);
    }
}

/* Deterministic, zero-mean-ish synthetic values.  A multiplicative hash keeps
 * the weights varied enough that the two KV groups disagree and that a broken
 * GQA grouping changes the output by far more than the tolerance. */
static float synth(uint32_t o, uint32_t i, float scale) {
    uint32_t v = (o * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
    v ^= v >> 13;
    v *= 2246822519u;
    v ^= v >> 16;
    const int32_t s = (int32_t)((v >> 8) % 2001u) - 1000;
    return scale * (float)s / 1000.0f;
}

static float rms_norm_one(const float *x, const float *weight, uint64_t n,
                          float *out) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1e-6f);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
    return scale;
}

/* Independent scalar reference for the whole layer core. */
static void reference_layer(const float *wq, const float *wk, const float *wv,
                            const float *wo, const float *norm_pre,
                            const float *norm_post, const float *x,
                            uint32_t n_tokens, float *out) {
    static float h[N_TOKENS * N_EMBD];
    static float q[N_TOKENS * Q_GATE_DIM];
    static float k[N_TOKENS * KV_DIM];
    static float v[N_TOKENS * KV_DIM];
    static float context[N_TOKENS * Q_DIM];
    static float proj[N_TOKENS * N_EMBD];
    static float resid[N_TOKENS * N_EMBD];

    for (uint32_t t = 0; t < n_tokens; t++) {
        rms_norm_one(x + (uint64_t)t * N_EMBD, norm_pre, N_EMBD,
                     h + (uint64_t)t * N_EMBD);
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *ht = h + (uint64_t)t * N_EMBD;
        float *qt = q + (uint64_t)t * Q_GATE_DIM;
        float *kt = k + (uint64_t)t * KV_DIM;
        float *vt = v + (uint64_t)t * KV_DIM;
        for (uint32_t o = 0; o < Q_GATE_DIM; o++) {
            double acc = 0.0;
            const float *row = wq + (uint64_t)o * N_EMBD;
            for (uint32_t i = 0; i < N_EMBD; i++) acc += (double)row[i] * ht[i];
            qt[o] = (float)acc;
        }
        for (uint32_t o = 0; o < KV_DIM; o++) {
            double acck = 0.0, accv = 0.0;
            const float *rk = wk + (uint64_t)o * N_EMBD;
            const float *rv = wv + (uint64_t)o * N_EMBD;
            for (uint32_t i = 0; i < N_EMBD; i++) {
                acck += (double)rk[i] * ht[i];
                accv += (double)rv[i] * ht[i];
            }
            kt[o] = (float)acck;
            vt[o] = (float)accv;
        }
    }

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t hd = 0; hd < N_HEAD; hd++) {
            const uint32_t kv_head = hd / GROUP;
            const float *qh = q + (uint64_t)t * Q_GATE_DIM + (uint64_t)hd * HEAD_DIM;
            float *ctx = context + (uint64_t)t * Q_DIM + (uint64_t)hd * HEAD_DIM;
            float scores[N_TOKENS];
            float max_score = -FLT_MAX;
            for (uint32_t s = 0; s <= t; s++) {
                const float *kh =
                    k + (uint64_t)s * KV_DIM + (uint64_t)kv_head * HEAD_DIM;
                double dot = 0.0;
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    dot += (double)qh[d] * kh[d];
                }
                scores[s] = (float)dot * scale;
                if (scores[s] > max_score) max_score = scores[s];
            }
            double denom = 0.0;
            for (uint32_t s = 0; s <= t; s++) {
                scores[s] = expf(scores[s] - max_score);
                denom += scores[s];
            }
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                double acc = 0.0;
                for (uint32_t s = 0; s <= t; s++) {
                    const float *vh =
                        v + (uint64_t)s * KV_DIM + (uint64_t)kv_head * HEAD_DIM;
                    acc += ((double)scores[s] / denom) * vh[d];
                }
                ctx[d] = (float)acc;
            }
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *ct = context + (uint64_t)t * Q_DIM;
        for (uint32_t o = 0; o < N_EMBD; o++) {
            double acc = 0.0;
            const float *row = wo + (uint64_t)o * Q_DIM;
            for (uint32_t j = 0; j < Q_DIM; j++) acc += (double)row[j] * ct[j];
            proj[(uint64_t)t * N_EMBD + o] = (float)acc;
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < N_EMBD; d++) {
            const uint64_t idx = (uint64_t)t * N_EMBD + d;
            resid[idx] = x[idx] + proj[idx];
        }
        rms_norm_one(resid + (uint64_t)t * N_EMBD, norm_post, N_EMBD,
                     out + (uint64_t)t * N_EMBD);
    }
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    float *wq = (float *)model;
    float *wk = wq + (uint64_t)Q_GATE_DIM * N_EMBD;
    float *wv = wk + (uint64_t)KV_DIM * N_EMBD;
    float *wo = wv + (uint64_t)KV_DIM * N_EMBD;
    float *norm_pre = wo + (uint64_t)N_EMBD * Q_DIM;
    float *norm_post = norm_pre + N_EMBD;
    float *q_norm = norm_post + N_EMBD;
    float *k_norm = q_norm + HEAD_DIM;
    require_ok((uint8_t *)(k_norm + HEAD_DIM) <= model + MODEL_BYTES,
               "synthetic weights fit the mapping");

    for (uint32_t o = 0; o < Q_GATE_DIM; o++)
        for (uint32_t i = 0; i < N_EMBD; i++)
            wq[(uint64_t)o * N_EMBD + i] = synth(o, i, 0.02f);
    for (uint32_t o = 0; o < KV_DIM; o++)
        for (uint32_t i = 0; i < N_EMBD; i++) {
            wk[(uint64_t)o * N_EMBD + i] = synth(o + 7u, i, 0.02f);
            wv[(uint64_t)o * N_EMBD + i] = synth(o + 23u, i, 0.02f);
        }
    for (uint32_t o = 0; o < N_EMBD; o++)
        for (uint32_t j = 0; j < Q_DIM; j++)
            wo[(uint64_t)o * Q_DIM + j] = synth(o + 101u, j, 0.02f);
    for (uint32_t d = 0; d < N_EMBD; d++) {
        norm_pre[d] = 0.9f + 0.0002f * (float)(d % 997u);
        norm_post[d] = 1.1f - 0.00015f * (float)(d % 991u);
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        q_norm[d] = 1.0f;
        k_norm[d] = 1.0f;
    }

    float x[N_TOKENS * N_EMBD];
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            x[(uint64_t)t * N_EMBD + d] =
                synth(d + 1u, t + 17u, 0.6f) + 0.01f * (float)t;

    float expected[N_TOKENS * N_EMBD];
    reference_layer(wq, wk, wv, wo, norm_pre, norm_post, x, N_TOKENS, expected);

    float actual[N_TOKENS * N_EMBD];
    ds4_test_qwen35_attn_args args = {
        .attn_q = wq,
        .attn_k = wk,
        .attn_v = wv,
        .attn_output = wo,
        .attn_norm = norm_pre,
        .ffn_norm = norm_post,
        .attn_q_norm = q_norm,
        .attn_k_norm = k_norm,
        .x = x,
        .out = actual,
        .n_tokens = N_TOKENS,
        .il = 3,
        .n_rot = 0,
        .pos0 = 0,
        .rope_freq_base = 10000000.0f,
    };

    require_ok(ds4_test_qwen35_attn_forward(&args) == 0, "attn forward (3 tokens)");
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 attn layer", actual[(uint64_t)t * N_EMBD + d],
                          expected[(uint64_t)t * N_EMBD + d], 1e-5f);

    /* The first two rows of a 3-token batch must equal a 1- and a 2-token run:
     * this pins causal masking (a later key must not influence an earlier
     * query) and the no-mixing single-token case. */
    for (uint32_t n = 1; n <= 2; n++) {
        float prefix[N_TOKENS * N_EMBD];
        args.n_tokens = n;
        args.out = prefix;
        require_ok(ds4_test_qwen35_attn_forward(&args) == 0,
                   n == 1 ? "attn forward (1 token)" : "attn forward (2 tokens)");
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t d = 0; d < N_EMBD; d++)
                require_close("qwen35 causal prefix",
                              prefix[(uint64_t)t * N_EMBD + d],
                              actual[(uint64_t)t * N_EMBD + d], 0.0f);
    }

    /* The two KV groups must actually differ; otherwise the GQA test above is
     * vacuous.  Compare the kv=0 and kv=1 K slices of the first projection. */
    float group_delta = 0.0f;
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        group_delta += fabsf(wk[d] - wk[(uint64_t)HEAD_DIM + d]);
    }
    require_ok(group_delta > 1.0f, "the two KV groups differ");

    /* il=3 is a full-attention block; il=4 is a linear (gated delta net) block
     * per ds4_qwen35moe_layer_is_linear.  The hook must refuse the latter: it
     * has no attn_q/k/v and is not the layer this core models. */
    args.il = 4;
    args.out = actual;
    require_ok(ds4_test_qwen35_attn_forward(&args) != 0,
               "attn forward rejects a linear layer");
    args.il = 3;

    munmap(model, MODEL_BYTES);
    puts("qwen35 full-attention layer core: PASS");
    return 0;
}
