/* Model-free host-side checks for dense qwen35 weights:
 *
 *  1. qwen4_ref_row's Q5_K and Q6_K cases (the 4B's tied Q6_K token
 *     embedding is gathered through it for every prompt token) against an
 *     independent, per-element ggml-order reference in double precision;
 *  2. qwen35_graph_weights_supported's quant-type refusals (Q5_K / Q6_K
 *     alpha and beta) with the sibling cases it must accept.
 *
 * ds4.c is included so the statics are reachable; the test links the Metal
 * objects but opens no device and no model. */
#include "../ds4.c"
#include <assert.h>

static int failures;

#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

/* Finite, non-negative half in a modest range, as raw bits. */
static uint16_t random_half_bits(void) {
    return (uint16_t)(((8u + rnd() % 8u) << 10) | (rnd() & 0x3ffu));
}

static double half_to_double(uint16_t h) {
    const int e = (h >> 10) & 0x1f;
    const double f = h & 0x3ff;
    const double v = e ? ldexp(1.0 + f / 1024.0, e - 15) : ldexp(f / 1024.0, -14);
    return (h & 0x8000u) ? -v : v;
}

/* ggml's 12-byte packed 6-bit scale and min of sub-block j (get_scale_min_k4). */
static void k4_scale_min(int j, const uint8_t *q, int *sc, int *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* Q5_K block: d, dmin (2+2), scales[12], qh[32], qs[128] = 176 bytes.
 * Element e is in sub-block g = e / 32: the low nibble of qs when g is even,
 * the high nibble when odd, read from qs[(g / 2) * 32 + e % 32]; its fifth
 * bit is bit g of qh[e % 32]. */
static double ref_q5_K(const uint8_t *b, int e, double *bound) {
    const double d = half_to_double((uint16_t)(b[0] | (b[1] << 8)));
    const double dmin = half_to_double((uint16_t)(b[2] | (b[3] << 8)));
    const int g = e / 32, l = e % 32;
    int sc, mn;
    k4_scale_min(g, b + 4, &sc, &mn);
    const uint8_t byte = b[48 + (g / 2) * 32 + l];
    const int low = (g & 1) ? (byte >> 4) : (byte & 0xF);
    const int q = low + (((b[16 + l] >> g) & 1) ? 16 : 0);
    *bound = fabs(d * sc * q) + fabs(dmin * mn);
    return d * sc * q - dmin * mn;
}

/* Q6_K block: ql[128], qh[64], scales[16] (int8), d (half) = 210 bytes.
 * Element e: half h = e / 128, quarter qd = (e % 128) / 32, lane l = e % 32.
 * The low 4 bits come from ql[64h + l + 32 (qd & 1)], low nibble for quarters
 * 0 and 1, high nibble for 2 and 3; the top 2 bits are
 * (qh[32h + l] >> 2 qd) & 3; the scale is scales[8h + l / 16 + 2 qd]. */
static double ref_q6_K(const uint8_t *b, int e, double *bound) {
    const double d = half_to_double((uint16_t)(b[208] | (b[209] << 8)));
    const int h = e / 128, qd = (e % 128) / 32, l = e % 32;
    const uint8_t byte = b[64 * h + l + 32 * (qd & 1)];
    const int low = qd >= 2 ? (byte >> 4) : (byte & 0xF);
    const int high = (b[128 + 32 * h + l] >> (2 * qd)) & 3;
    const int sc = (int8_t)b[192 + 8 * h + l / 16 + 2 * qd];
    const int q = (low | (high << 4)) - 32;
    *bound = fabs(d * sc * q);
    return d * sc * q;
}

static void test_ref_row(uint32_t type, uint64_t block_bytes, const char *name,
                         double (*ref)(const uint8_t *, int, double *)) {
    enum { COLS = 512, ROWS = 5 };
    const uint64_t blocks = COLS / 256u;
    const uint64_t row_bytes = blocks * block_bytes;
    uint8_t *data = malloc(ROWS * row_bytes);
    assert(data);
    for (uint64_t i = 0; i < ROWS * row_bytes; i++) data[i] = (uint8_t)rnd();
    /* well-formed d (and dmin): a finite half in each block */
    for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t k = 0; k < blocks; k++) {
            uint8_t *b = data + r * row_bytes + k * block_bytes;
            if (type == DS4_TENSOR_Q6_K) {
                const uint16_t d = random_half_bits();
                b[208] = (uint8_t)d; b[209] = (uint8_t)(d >> 8);
            } else {
                const uint16_t d = random_half_bits(), dm = random_half_bits();
                b[0] = (uint8_t)d; b[1] = (uint8_t)(d >> 8);
                b[2] = (uint8_t)dm; b[3] = (uint8_t)(dm >> 8);
            }
        }
    }
    ds4_model m;
    memset(&m, 0, sizeof(m));
    m.map = data;
    m.size = ROWS * row_bytes;
    ds4_tensor t;
    memset(&t, 0, sizeof(t));
    t.ndim = 2; t.dim[0] = COLS; t.dim[1] = ROWS;
    t.type = type; t.abs_offset = 0; t.bytes = ROWS * row_bytes; t.elements = (uint64_t)COLS * ROWS;

    float out[COLS];
    int bad = 0;
    double worst = 0.0;
    for (uint64_t r = 0; r < ROWS; r++) {
        qwen4_ref_row(&m, &t, r, out);
        for (int e = 0; e < COLS; e++) {
            double bound;
            const uint8_t *b = data + r * row_bytes + (uint64_t)(e / 256) * block_bytes;
            const double want = ref(b, e % 256, &bound);
            const double err = fabs((double)out[e] - want);
            const double tol = 8.0 * FLT_EPSILON * bound + 1e-30;
            if (err > tol) {
                if (bad < 4) printf("  %s row %llu col %d: got %.9g want %.9g\n", name,
                                    (unsigned long long)r, e, (double)out[e], want);
                bad++;
            }
            if (bound > 0 && err / bound > worst) worst = err / bound;
        }
    }
    CHECK(bad == 0, "%s qwen4_ref_row: %d of %d elements differ from the ggml-order reference", name, bad, COLS * ROWS);
    printf("qwen4_ref_row %s: %d elements vs double reference, %d bad, worst relative error %.2e\n",
           name, COLS * ROWS, bad, worst);
    free(data);
}

/* --- qwen35_graph_weights_supported ------------------------------------- */

static ds4_tensor g_t[DS4_TENSOR_MXFP4 + 1];

static ds4_tensor *tensor_of(uint32_t type) {
    ds4_tensor *t = &g_t[type];
    memset(t, 0, sizeof(*t));
    t->ndim = 2; t->dim[0] = 512; t->dim[1] = 64; t->type = type;
    return t;
}

typedef struct {
    uint32_t token_embd, output, qkv, alpha, beta;
} wcase;

static bool run_case(const wcase *c) {
    /* three GDN layers and one attention layer, no MTP block */
    g_ds4_shape = DS4_SHAPE_QWEN35;
    g_ds4_shape.n_layer = 4;
    g_ds4_shape.n_nextn_predict = 0;
    g_ds4_shape.n_full_attn_interval = 4;
    ds4_weights *w = calloc(1, sizeof(*w));
    assert(w);
    static ds4_tensor norm, te, out, qkv, alpha, beta, f16;
    te = *tensor_of(c->token_embd);
    out = *tensor_of(c->output);
    qkv = *tensor_of(c->qkv);
    alpha = *tensor_of(c->alpha);
    beta = *tensor_of(c->beta);
    f16 = *tensor_of(DS4_TENSOR_F16);
    memset(&norm, 0, sizeof(norm));
    w->token_embd = &te; w->output = &out; w->output_norm = &norm;
    for (uint32_t il = 0; il < 4; il++) {
        ds4_layer_weights *l = &w->layer[il];
        l->ffn_gate = l->ffn_up = l->ffn_down = &f16;
        if (ds4_qwen4_layer_is_linear(il)) {
            l->lin_qkv = &qkv; l->lin_gate = &f16; l->lin_out = &f16;
            l->lin_alpha = &alpha; l->lin_beta = &beta;
        } else {
            l->attn_q = l->attn_k = l->attn_v = l->attn_output = &f16;
        }
    }
    const bool ok = qwen35_graph_weights_supported(w);
    free(w);
    return ok;
}

static void test_weights_supported(void) {
    enum { F16 = DS4_TENSOR_F16, F32 = DS4_TENSOR_F32, Q4K = DS4_TENSOR_Q4_K, Q5K = DS4_TENSOR_Q5_K,
           Q6K = DS4_TENSOR_Q6_K, Q80 = DS4_TENSOR_Q8_0 };
    /* the success siblings: every quant the dense projections may carry */
    const wcase accept[] = {
        {Q4K, Q4K, Q4K, F16, F16},   /* baseline */
        {Q4K, Q4K, Q5K, F16, F16},   /* Q5_K lin_qkv */
        {Q4K, Q4K, Q6K, F16, F16},   /* Q6_K lin_qkv */
        {Q6K, Q4K, Q4K, F16, F16},   /* Q6_K token_embd (the 4B) */
        {Q6K, Q6K, Q5K, F32, F32},   /* tied Q6_K head, f32 alpha/beta */
        {Q4K, Q4K, Q4K, Q4K, Q4K},   /* matching Q4_K alpha/beta */
        {Q4K, Q4K, Q4K, Q80, Q80},   /* matching Q8_0 alpha/beta */
    };
    for (size_t i = 0; i < sizeof(accept) / sizeof(accept[0]); i++)
        CHECK(run_case(&accept[i]), "accept case %zu (embd=%u out=%u qkv=%u alpha=%u beta=%u) was declined", i,
              accept[i].token_embd, accept[i].output, accept[i].qkv, accept[i].alpha, accept[i].beta);
    /* the refusals: alpha/beta the fused decode front cannot read */
    const wcase decline[] = {
        {Q4K, Q4K, Q4K, Q5K, Q5K},   /* Q5_K alpha and beta */
        {Q4K, Q4K, Q4K, Q6K, Q6K},   /* Q6_K alpha and beta */
        {Q6K, Q6K, Q6K, Q6K, Q6K},   /* ... even with a Q6_K model throughout */
        {Q4K, Q4K, Q4K, F16, F32},   /* mismatched alpha/beta */
        {Q4K, Q4K, Q4K, Q5K, F16},   /* Q5_K alpha with an f16 beta */
        {Q4K, Q4K, Q4K, F16, Q6K},   /* Q6_K beta with an f16 alpha */
    };
    for (size_t i = 0; i < sizeof(decline) / sizeof(decline[0]); i++)
        CHECK(!run_case(&decline[i]), "decline case %zu (alpha=%u beta=%u) was accepted", i,
              decline[i].alpha, decline[i].beta);
}

int main(void) {
    test_ref_row(DS4_TENSOR_Q5_K, 176, "Q5_K", ref_q5_K);
    test_ref_row(DS4_TENSOR_Q6_K, 210, "Q6_K", ref_q6_K);
    test_weights_supported();
    if (failures) { printf("%d failure(s)\n", failures); return 1; }
    printf("all qwen35 host tests passed\n");
    return 0;
}
