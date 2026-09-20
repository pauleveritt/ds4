#include "../ds4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int id;
    float logit;
    float prob;
} reference_candidate;

static int failures;

#define CHECK(cond, ...) do {                                                 \
    if (!(cond)) {                                                            \
        fprintf(stderr, "FAIL: " __VA_ARGS__);                               \
        fputc('\n', stderr);                                                  \
        failures++;                                                           \
    }                                                                         \
} while (0)

static uint64_t reference_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float reference_rng_f32(uint64_t *state) {
    const uint64_t x = reference_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int reference_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = (int)i;
        }
    }
    return best;
}

static int reference_candidate_cmp_desc(const void *a, const void *b) {
    const reference_candidate *ca = a;
    const reference_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

/* This is the sampler immediately before the optimized implementation. */
static int reference_sample(const float *logits, uint32_t n_vocab,
                            float temperature, int top_k,
                            float top_p, float min_p, uint64_t *rng) {
    if (temperature <= 0.0f) return reference_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;

    if (top_k > 0) {
        if (top_k > 1024) top_k = 1024;
        if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;
        int ids[1024];
        float vals[1024];
        int n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == top_k && v <= vals[n - 1]) continue;
            int j = n < top_k ? n++ : n - 1;
            while (j > 0 && vals[j - 1] < v) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            vals[j] = v;
            ids[j] = (int)i;
        }
        if (n == 0) return reference_argmax(logits, n_vocab);

        float probs[1024];
        const float max_logit = vals[0];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            probs[i] = expf((vals[i] - max_logit) / temperature);
            sum += probs[i];
        }
        if (sum <= 0.0f || !isfinite(sum)) return ids[0];
        const float min_prob = (probs[0] / sum) * min_p;
        float filtered_sum = 0.0f;
        int filtered = 0;
        for (int i = 0; i < n; i++) {
            const float p = probs[i] / sum;
            if (i > 0 && p < min_prob) break;
            filtered_sum += probs[i];
            filtered++;
            if (filtered_sum / sum >= top_p) break;
        }
        float r = reference_rng_f32(rng) * filtered_sum;
        for (int i = 0; i < filtered; i++) {
            r -= probs[i];
            if (r <= 0.0f) return ids[i];
        }
        return ids[filtered - 1];
    }

    float max_logit = -INFINITY;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return reference_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = reference_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    reference_candidate *cand = malloc((size_t)finite * sizeof(*cand));
    CHECK(cand != NULL, "reference candidate allocation");
    if (!cand) return best;
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (reference_candidate){(int)i, v, p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }
    qsort(cand, n, sizeof(*cand), reference_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    float r = reference_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static uint64_t data_rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

static void fill_logits(float *logits, uint32_t n, uint64_t seed) {
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t x = data_rng_next(&seed);
        const int32_t q = (int32_t)(x >> 32) % 250000;
        logits[i] = (float)q / 10000.0f + (float)i * 1.0e-7f;
    }
    if (n > 17) logits[17] = -INFINITY;
    if (n > 113) logits[113] = NAN;
}

static void compare_case(const float *logits, float *scratch, uint32_t n,
                         float temperature, int top_k,
                         float top_p, float min_p, const char *label) {
    for (uint64_t seed = 0; seed < 256; seed++) {
        uint64_t ref_rng = seed;
        uint64_t opt_rng = seed;
        const int ref = reference_sample(logits, n, temperature, top_k,
                                         top_p, min_p, &ref_rng);
        const int opt = ds4_test_sample_logits(logits, n, temperature, top_k,
                                               top_p, min_p, &opt_rng, scratch);
        CHECK(ref == opt,
              "%s seed=%llu token reference=%d optimized=%d",
              label, (unsigned long long)seed, ref, opt);
        CHECK(ref_rng == opt_rng,
              "%s seed=%llu RNG reference=%llu optimized=%llu",
              label, (unsigned long long)seed,
              (unsigned long long)ref_rng, (unsigned long long)opt_rng);
    }
}

static void check_greedy_argmax_case(const float *logits, uint32_t n,
                                     int expected, const char *label) {
    uint64_t unrolled_rng = 0x1234u;
    uint64_t scalar_rng = unrolled_rng;
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s select unrolled argmax", label);
    const int unrolled = ds4_test_sample_logits(
            logits, n, 0.0f, 0, 1.0f, 0.0f, &unrolled_rng, NULL);
    CHECK(setenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX", "1", 1) == 0,
          "%s select scalar argmax", label);
    const int scalar = ds4_test_sample_logits(
            logits, n, 0.0f, 0, 1.0f, 0.0f, &scalar_rng, NULL);
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s restore unrolled argmax", label);
    CHECK(unrolled == scalar,
          "%s unrolled=%d scalar=%d", label, unrolled, scalar);
    CHECK(unrolled == expected,
          "%s token=%d expected=%d", label, unrolled, expected);
    CHECK(unrolled_rng == 0x1234u && scalar_rng == 0x1234u,
          "%s greedy argmax changed RNG state", label);
}

static void check_excluding_argmax_case(const float *logits, uint32_t n,
                                        int excluded_id, int expected,
                                        const char *label) {
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s select unrolled excluding argmax", label);
    const int unrolled = ds4_test_argmax_excluding_logits(
            logits, n, excluded_id);
    CHECK(setenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX", "1", 1) == 0,
          "%s select scalar excluding argmax", label);
    const int scalar = ds4_test_argmax_excluding_logits(
            logits, n, excluded_id);
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s restore unrolled excluding argmax", label);
    CHECK(unrolled == scalar,
          "%s excluded=%d unrolled=%d scalar=%d",
          label, excluded_id, unrolled, scalar);
    CHECK(unrolled == expected,
          "%s excluded=%d token=%d expected=%d",
          label, excluded_id, unrolled, expected);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void check_speculative_distribution(void) {
    const float filter_logits[] = {
        logf(0.50f), logf(0.30f), logf(0.15f), logf(0.05f),
    };
    float probs[4];
    CHECK(ds4_test_sampling_probabilities(filter_logits, 4, 1.0f,
                                          3, 0.70f, 0.0f, probs) != 0,
          "build top-k/top-p probabilities");
    CHECK(fabsf(probs[0] - 0.625f) < 1e-6f,
          "top-k/top-p p0 %.9g", probs[0]);
    CHECK(fabsf(probs[1] - 0.375f) < 1e-6f,
          "top-k/top-p p1 %.9g", probs[1]);
    CHECK(probs[2] == 0.0f && probs[3] == 0.0f,
          "top-k/top-p filtered tail %.9g %.9g", probs[2], probs[3]);

    CHECK(ds4_test_sampling_probabilities(filter_logits, 4, 1.0f,
                                          0, 1.0f, 0.40f, probs) != 0,
          "build min-p probabilities");
    CHECK(fabsf(probs[0] - 0.625f) < 1e-6f,
          "min-p p0 %.9g", probs[0]);
    CHECK(fabsf(probs[1] - 0.375f) < 1e-6f,
          "min-p p1 %.9g", probs[1]);
    CHECK(probs[2] == 0.0f && probs[3] == 0.0f,
          "min-p filtered tail %.9g %.9g", probs[2], probs[3]);

    const float target_logits[] = {
        logf(0.62f), logf(0.27f), logf(0.11f),
    };
    const float draft_logits[] = {
        logf(0.15f), logf(0.55f), logf(0.30f),
    };
    uint64_t rng = 0x45f17a9d2c6b0381ULL;
    uint32_t counts[3] = {0};
    float target_probs[3];
    float draft_probs[3];
    const uint32_t trials = 100000;
    for (uint32_t i = 0; i < trials; i++) {
        const int token = ds4_test_speculative_sample(
            target_logits, draft_logits, 3, 1.0f, 0, 1.0f, 0.0f,
            &rng, target_probs, draft_probs);
        CHECK(token >= 0 && token < 3,
              "speculative sample token %d", token);
        if (token >= 0 && token < 3) counts[token]++;
    }
    const float expected[] = {0.62f, 0.27f, 0.11f};
    for (uint32_t i = 0; i < 3; i++) {
        const float observed = (float)counts[i] / (float)trials;
        CHECK(fabsf(observed - expected[i]) < 0.006f,
              "speculative distribution token=%u observed=%.6f expected=%.6f",
              i, observed, expected[i]);
    }
    printf("stochastic speculative distribution: %.4f %.4f %.4f\n",
           (double)counts[0] / trials,
           (double)counts[1] / trials,
           (double)counts[2] / trials);

    memset(counts, 0, sizeof(counts));
    rng = 0x8f76c2b5a149d30eULL;
    for (uint32_t i = 0; i < trials; i++) {
        const int token = ds4_test_speculative_delta_sample(
            target_logits, 3, 0, 1.0f, 0, 1.0f, 0.0f,
            &rng, target_probs);
        CHECK(token >= 0 && token < 3,
              "delta speculative sample token %d", token);
        if (token >= 0 && token < 3) counts[token]++;
    }
    for (uint32_t i = 0; i < 3; i++) {
        const float observed = (float)counts[i] / (float)trials;
        CHECK(fabsf(observed - expected[i]) < 0.006f,
              "delta distribution token=%u observed=%.6f expected=%.6f",
              i, observed, expected[i]);
    }
    printf("delta speculative distribution: %.4f %.4f %.4f\n",
           (double)counts[0] / trials,
           (double)counts[1] / trials,
           (double)counts[2] / trials);
}

/* ds4_engine_sampling_defaults() is the single source of sampler defaults
 * (no Rust or Python table may duplicate it), so each model family's answer
 * is pinned here.  Model-free: the hook selects the shape, not a GGUF. */
static void check_sampling_defaults_case(const char *name,
                                         ds4_variant variant,
                                         float temperature, int top_k,
                                         float top_p, float min_p) {
    float t = -1.0f, p = -1.0f, m = -1.0f;
    int k = -1;
    CHECK(ds4_test_sampling_defaults_for_variant(variant, &t, &k, &p, &m) == 0,
          "%s: sampling-defaults hook refused variant %d", name, (int)variant);
    printf("sampling defaults %s: temperature=%.9g top_k=%d top_p=%.9g "
           "min_p=%.9g\n", name, (double)t, k, (double)p, (double)m);
    CHECK(t == temperature, "%s temperature %.9g != %.9g",
          name, (double)t, (double)temperature);
    CHECK(k == top_k, "%s top_k %d != %d", name, k, top_k);
    CHECK(p == top_p, "%s top_p %.9g != %.9g",
          name, (double)p, (double)top_p);
    CHECK(m == min_p, "%s min_p %.9g != %.9g",
          name, (double)m, (double)min_p);
}

static void check_sampling_defaults(void) {
    /* Flash (DeepSeek4): the generic defaults, unchanged. */
    check_sampling_defaults_case("flash", DS4_VARIANT_FLASH,
                                 DS4_DEFAULT_TEMPERATURE, 0,
                                 DS4_DEFAULT_TOP_P, DS4_DEFAULT_MIN_P);
    /* Laguna XS 2.1: top_k 20, min_p 0, generic temperature and top_p. */
    check_sampling_defaults_case("laguna-xs21", DS4_VARIANT_LAGUNA_XS21,
                                 DS4_DEFAULT_TEMPERATURE, 20,
                                 DS4_DEFAULT_TOP_P, 0.0f);
    /* Mellum 2: the public model card's Quickstart values (revision
     * a7311550557e93cc706ab5dd3d879c1a11703ab4, README.md:222-225). */
    check_sampling_defaults_case("mellum2", DS4_VARIANT_MELLUM2,
                                 0.6f, 20, 0.95f, 0.0f);
}

/*
 * Mellum memory accounting (P20).  The engine plans weights + KV + scratch at
 * open and refuses the session when the plan exceeds the device's recommended
 * working set.  The arithmetic is shared with the allocators, so it is pinned
 * here against literals derived BY HAND from the Mellum 2 shape
 * (DS4_SHAPE_MELLUM2: n_layer 28, n_embd 2304, n_vocab 98304, n_head 32,
 * n_head_kv 4, n_head_dim 128, n_expert 64, n_expert_used 8, n_ff_exp 896,
 * n_swa 1024) and the allocation list in ds4_mellum_decode_state_create /
 * ds4_mellum_decode_output_prepare / ds4_mellum_prefill_scratch_create.
 * Model-free: the hook selects the shape, not a GGUF.
 */

/* kv_dim = n_head_kv 4 * n_head_dim 128 = 512 elements, 2 bytes each.
 * ds4_mellum_layer_uses_sliding_attention is (il & 3) != 3, so of 28 layers
 * 21 slide (cap = n_swa 1024) and 7 are full (cap = ctx).  K and V each.
 *   21 * 2 * (1024 * 512 * 2) = 44,040,192
 * +  7 * 2 * (4096 * 512 * 2) = 58,720,256
 *                             = 102,760,448 */
#define MELLUM_KV_BYTES_CTX4096 UINT64_C(102760448)
/* ctx 1024: every layer's cap is 1024, so 28 * 2 * (1024 * 512 * 2). */
#define MELLUM_KV_BYTES_CTX1024 UINT64_C(58720256)

/* Decode-state GPU tensors other than the caches and the output head
 * (ds4_mellum_decode_state_create):
 *   7 x embd   (hidden, layer_out, attention_out, attention_norm, projected,
 *               ffn_norm, moe_out)  7 * 2304 * 4          =  64,512
 *   2 x q      (q, heads)           2 * 32 * 128 * 4      =  32,768
 *   2 x kv     (k, v)               2 * 4 * 128 * 4       =   4,096
 *   2 x expert (router_logits, router_probs) 2 * 64 * 4   =     512
 *   router_selected                 8 * 4                 =      32
 *   router_weights                  8 * 4                 =      32
 *   moe_mid                         8 * 896 * 4           =  28,672
 *                                                         = 130,624
 * hidden_cpu is a host malloc, not a GPU tensor, and is not counted. */
#define MELLUM_DECODE_SCRATCH_BYTES UINT64_C(130624)
/* Output head (ds4_mellum_decode_output_prepare):
 *   output_norm  2304 * 4    =   9,216
 *   logits      98304 * 4    = 393,216
 *                            = 402,432 */
#define MELLUM_DECODE_OUTPUT_BYTES UINT64_C(402432)

/* Prefill scratch per token (ds4_mellum_prefill_scratch_create):
 *   tokens                 4          staged_key   4*128*2 =  1,024
 *   7 x embd        64,512            staged_value         =  1,024
 *   2 x q (q,heads) 32,768            2 x expert f32       =    512
 *   2 x kv (k,v)     4,096            router_selected      =     32
 *   moe_mid         28,672            router_weights       =     32
 * total per token = 132,676 */
#define MELLUM_PREFILL_BYTES_PER_TOKEN UINT64_C(132676)

static void check_mellum_memory_plan(void) {
    ds4_test_mellum_memory m;
    memset(&m, 0, sizeof(m));
    /* ctx 4096, no prefill scratch (the product session allocates none), a
     * round weights figure so the sum is checkable by eye. */
    const uint64_t weights = UINT64_C(12000000000);
    CHECK(ds4_test_mellum_memory_plan(4096, 0, weights, &m) == 0,
          "mellum memory-plan hook refused ctx 4096");
    printf("mellum plan ctx=4096: kv=%llu decode_scratch=%llu output=%llu "
           "state=%llu prefill=%llu planned=%llu\n",
           (unsigned long long)m.kv_bytes,
           (unsigned long long)m.decode_scratch_bytes,
           (unsigned long long)m.decode_output_bytes,
           (unsigned long long)m.decode_state_bytes,
           (unsigned long long)m.prefill_scratch_bytes,
           (unsigned long long)m.planned_bytes);
    CHECK(m.kv_bytes == MELLUM_KV_BYTES_CTX4096,
          "mellum kv(4096) %llu != %llu",
          (unsigned long long)m.kv_bytes,
          (unsigned long long)MELLUM_KV_BYTES_CTX4096);
    CHECK(m.decode_scratch_bytes == MELLUM_DECODE_SCRATCH_BYTES,
          "mellum decode scratch %llu != %llu",
          (unsigned long long)m.decode_scratch_bytes,
          (unsigned long long)MELLUM_DECODE_SCRATCH_BYTES);
    CHECK(m.decode_output_bytes == MELLUM_DECODE_OUTPUT_BYTES,
          "mellum decode output %llu != %llu",
          (unsigned long long)m.decode_output_bytes,
          (unsigned long long)MELLUM_DECODE_OUTPUT_BYTES);
    CHECK(m.decode_state_bytes == MELLUM_KV_BYTES_CTX4096 +
                                  MELLUM_DECODE_SCRATCH_BYTES +
                                  MELLUM_DECODE_OUTPUT_BYTES,
          "mellum decode state %llu != %llu",
          (unsigned long long)m.decode_state_bytes,
          (unsigned long long)(MELLUM_KV_BYTES_CTX4096 +
                               MELLUM_DECODE_SCRATCH_BYTES +
                               MELLUM_DECODE_OUTPUT_BYTES));
    CHECK(m.prefill_scratch_bytes == 0, "mellum prefill(0) %llu != 0",
          (unsigned long long)m.prefill_scratch_bytes);
    CHECK(m.planned_bytes == weights + m.decode_state_bytes,
          "mellum planned %llu != %llu",
          (unsigned long long)m.planned_bytes,
          (unsigned long long)(weights + m.decode_state_bytes));

    /* ctx 1024: no layer exceeds the sliding window, so every cap is 1024. */
    memset(&m, 0, sizeof(m));
    CHECK(ds4_test_mellum_memory_plan(1024, 0, 0, &m) == 0,
          "mellum memory-plan hook refused ctx 1024");
    CHECK(m.kv_bytes == MELLUM_KV_BYTES_CTX1024,
          "mellum kv(1024) %llu != %llu",
          (unsigned long long)m.kv_bytes,
          (unsigned long long)MELLUM_KV_BYTES_CTX1024);

    /* Prefill scratch is linear in cap and allocated only by the probes. */
    memset(&m, 0, sizeof(m));
    CHECK(ds4_test_mellum_memory_plan(1024, 1, 0, &m) == 0,
          "mellum memory-plan hook refused prefill cap 1");
    CHECK(m.prefill_scratch_bytes == MELLUM_PREFILL_BYTES_PER_TOKEN,
          "mellum prefill(1) %llu != %llu",
          (unsigned long long)m.prefill_scratch_bytes,
          (unsigned long long)MELLUM_PREFILL_BYTES_PER_TOKEN);
    memset(&m, 0, sizeof(m));
    CHECK(ds4_test_mellum_memory_plan(1024, 32, 0, &m) == 0,
          "mellum memory-plan hook refused prefill cap 32");
    CHECK(m.prefill_scratch_bytes == 32u * MELLUM_PREFILL_BYTES_PER_TOKEN,
          "mellum prefill(32) %llu != %llu",
          (unsigned long long)m.prefill_scratch_bytes,
          (unsigned long long)(32u * MELLUM_PREFILL_BYTES_PER_TOKEN));
    printf("mellum prefill scratch: cap1=%llu cap32=%llu\n",
           (unsigned long long)MELLUM_PREFILL_BYTES_PER_TOKEN,
           (unsigned long long)(32u * MELLUM_PREFILL_BYTES_PER_TOKEN));

    /* Admission truth table: a zero budget means the device did not answer,
     * so the plan is print-only and admits. */
    CHECK(ds4_test_mellum_admit(100, 100) == 1, "admit: equal must admit");
    CHECK(ds4_test_mellum_admit(99, 100) == 1, "admit: under must admit");
    CHECK(ds4_test_mellum_admit(101, 100) == 0, "admit: over must refuse");
    CHECK(ds4_test_mellum_admit(0, 0) == 1, "admit: unknown budget admits");
    CHECK(ds4_test_mellum_admit(UINT64_MAX, 0) == 1,
          "admit: unknown budget admits any plan");
    printf("mellum admit truth table: ok\n");
}

int main(void) {
    check_mellum_memory_plan();
    check_sampling_defaults();
    check_speculative_distribution();
    const uint32_t semantic_n = 4096;
    float *logits = malloc((size_t)semantic_n * sizeof(*logits));
    float *scratch = malloc((size_t)semantic_n * sizeof(*scratch));
    CHECK(logits && scratch, "semantic scratch allocation");
    if (!logits || !scratch) return 1;
    fill_logits(logits, semantic_n, 0x123456789abcdef0ULL);

    compare_case(logits, scratch, semantic_n, 1.0f, 0, 1.0f, 0.05f,
                 "default-min-p");
    compare_case(logits, scratch, semantic_n, 0.7f, 0, 1.0f, 0.0f,
                 "full-softmax");
    compare_case(logits, scratch, semantic_n, 1.3f, 0, 0.9f, 0.05f,
                 "top-p-min-p");
    compare_case(logits, scratch, semantic_n, 0.9f, 0, 0.95f, 0.0f,
                 "top-p");
    compare_case(logits, scratch, semantic_n, 0.8f, 64, 0.9f, 0.05f,
                 "top-k");
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "select unrolled argmax default");
    compare_case(logits, scratch, semantic_n, 0.0f, 0, 1.0f, 0.05f,
                 "greedy");
    CHECK(setenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX", "1", 1) == 0,
          "set scalar argmax control");
    compare_case(logits, scratch, semantic_n, 0.0f, 0, 1.0f, 0.05f,
                 "greedy-scalar-control");
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "restore unrolled argmax default");

    const float cross_lane_tie[] = {
        -4.0f, 9.0f, -2.0f, 3.0f, 1.0f, 5.0f, 0.0f, 7.0f,
         9.0f, 4.0f,  6.0f, 2.0f, 8.0f, 1.0f, 3.0f, 0.0f,
         9.0f,
    };
    check_greedy_argmax_case(
            cross_lane_tie,
            (uint32_t)(sizeof(cross_lane_tie) / sizeof(cross_lane_tie[0])),
            1, "greedy-cross-lane-tie");

    const float tail_tie[] = {
        -3.0f, 0.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
         6.0f, 7.0f, 8.0f,
    };
    check_greedy_argmax_case(
            tail_tie,
            (uint32_t)(sizeof(tail_tie) / sizeof(tail_tie[0])),
            2, "greedy-tail-tie");

    const float nonfinite_greedy[] = {
        NAN, -INFINITY, -2.0e30f, INFINITY, INFINITY, 1.0f, NAN, 0.0f, 2.0f,
    };
    check_greedy_argmax_case(
            nonfinite_greedy,
            (uint32_t)(sizeof(nonfinite_greedy) /
                       sizeof(nonfinite_greedy[0])),
            3, "greedy-nonfinite");

    const float below_sentinel[] = {
        -2.0e30f, -1.5e30f, -INFINITY, NAN, -3.0e30f,
    };
    check_greedy_argmax_case(
            below_sentinel,
            (uint32_t)(sizeof(below_sentinel) / sizeof(below_sentinel[0])),
            0, "greedy-below-sentinel");

    const uint32_t excluding_n =
        (uint32_t)(sizeof(cross_lane_tie) / sizeof(cross_lane_tie[0]));
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, -1, 1, "excluding-none");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, 1, 8, "excluding-best");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, 8, 1, "excluding-later-tie");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, 16, 1, "excluding-tail-tie");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, (int)excluding_n, 1,
            "excluding-out-of-range");
    check_excluding_argmax_case(
            nonfinite_greedy,
            (uint32_t)(sizeof(nonfinite_greedy) /
                       sizeof(nonfinite_greedy[0])),
            -1, 0, "excluding-leading-nan");
    check_excluding_argmax_case(
            below_sentinel,
            (uint32_t)(sizeof(below_sentinel) / sizeof(below_sentinel[0])),
            0, 1, "excluding-first-anchor");

    /* Exercise min-p values immediately around expf's cutoff. */
    const float cutoff = logf(0.05f);
    const float boundary_logits[] = {
        0.0f,
        cutoff,
        nextafterf(cutoff, -INFINITY),
        nextafterf(cutoff, INFINITY),
        -1.0f,
        -5.0f,
        -INFINITY,
        NAN,
    };
    float boundary_scratch[sizeof(boundary_logits) / sizeof(boundary_logits[0])];
    compare_case(boundary_logits, boundary_scratch,
                 (uint32_t)(sizeof(boundary_logits) / sizeof(boundary_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "min-p-boundary");

    const float tied_logits[] = {
        2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 0.0f, -INFINITY, NAN,
    };
    float tied_scratch[sizeof(tied_logits) / sizeof(tied_logits[0])];
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "equal-logits-default");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 0.8f, 0.05f, "equal-logits-top-p");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 0.01f, 0, 1.0f, 0.05f, "low-temperature");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 100.0f, 0, 1.0f, 0.05f, "high-temperature");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 1.0f, 1.0f, "min-p-one");

    uint64_t null_ref_rng = 42;
    uint64_t null_opt_rng = 42;
    const int null_ref = reference_sample(
            tied_logits,
            (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
            1.0f, 0, 1.0f, 0.05f, &null_ref_rng);
    const int null_opt = ds4_test_sample_logits(
            tied_logits,
            (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
            1.0f, 0, 1.0f, 0.05f, &null_opt_rng, NULL);
    CHECK(null_ref == null_opt && null_ref_rng == null_opt_rng,
          "null probability scratch fallback mismatch");

    const float nonfinite_logits[] = {-INFINITY, NAN, INFINITY, NAN};
    float nonfinite_scratch[sizeof(nonfinite_logits) /
                            sizeof(nonfinite_logits[0])];
    compare_case(nonfinite_logits, nonfinite_scratch,
                 (uint32_t)(sizeof(nonfinite_logits) /
                            sizeof(nonfinite_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "all-nonfinite");

    free(logits);
    free(scratch);

    const uint32_t perf_n = 163840;
    const int iterations = 100;
    logits = malloc((size_t)perf_n * sizeof(*logits));
    scratch = malloc((size_t)perf_n * sizeof(*scratch));
    CHECK(logits && scratch, "performance scratch allocation");
    if (!logits || !scratch) return 1;
    fill_logits(logits, perf_n, 0xfeedfacecafebeefULL);

    uint64_t ref_rng = 1234;
    uint64_t checksum = 0;
    double start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)reference_sample(logits, perf_n, 1.0f, 0,
                                               1.0f, 0.05f, &ref_rng);
    }
    const double reference_ms = (now_sec() - start) * 1000.0;

    uint64_t opt_rng = 1234;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)ds4_test_sample_logits(logits, perf_n, 1.0f, 0,
                                                     1.0f, 0.05f,
                                                     &opt_rng, scratch);
    }
    const double optimized_ms = (now_sec() - start) * 1000.0;
    CHECK(ref_rng == opt_rng, "performance RNG state mismatch");
    printf("sampling default: reference %.3f ms, optimized %.3f ms, %.2fx, checksum=%llu\n",
           reference_ms, optimized_ms,
           optimized_ms > 0.0 ? reference_ms / optimized_ms : 0.0,
           (unsigned long long)checksum);

    ref_rng = 5678;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)reference_sample(logits, perf_n, 1.0f, 0,
                                               0.9f, 0.05f, &ref_rng);
    }
    const double reference_top_p_ms = (now_sec() - start) * 1000.0;

    opt_rng = 5678;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)ds4_test_sample_logits(logits, perf_n, 1.0f, 0,
                                                     0.9f, 0.05f,
                                                     &opt_rng, scratch);
    }
    const double optimized_top_p_ms = (now_sec() - start) * 1000.0;
    CHECK(ref_rng == opt_rng, "top-p performance RNG state mismatch");
    printf("sampling top-p+min-p: reference %.3f ms, optimized %.3f ms, %.2fx, checksum=%llu\n",
           reference_top_p_ms, optimized_top_p_ms,
           optimized_top_p_ms > 0.0 ? reference_top_p_ms / optimized_top_p_ms : 0.0,
           (unsigned long long)checksum);

    free(logits);
    free(scratch);
    if (failures) {
        fprintf(stderr, "%d sampling test(s) failed\n", failures);
        return 1;
    }
    puts("sampling tests: OK");
    return 0;
}
