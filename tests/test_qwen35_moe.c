/* Model-free test for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE) MoE
 * feed-forward CPU layer core.
 *
 * The layer is exposed by a test hook that ds4.c compiles under
 * -DDS4_TEST_HOOKS.  The hook is not declared in ds4.h; the contract is
 * duplicated here, exactly as tests/test_qwen35_attn.c and
 * tests/test_qwen35_gdn.c duplicate their hooks.  Synthetic f32 weights live
 * in one anonymous mmap and are handed to the hook as explicit pointers, so the
 * test measures the layer and not the quantizer.
 *
 * The reference below is a plain scalar implementation of the same math: the
 * router projection (ffn_gate_inp . x), routing via the already-tested
 * ds4_test_qwen35_moe_route, the per-expert gate/up/down matmuls with the silu
 * SwiGLU, the weighted sum, and the sigmoid-gated shared expert via
 * ds4_test_qwen35_shared_expert_gate.  Routing is the shipped router's job, so
 * the reference consumes it rather than reimplementing it.
 *
 * Expert layout (the engine's tensor_expert_bytes, ds4.c:9031-9051): an expert
 * stack is a contiguous [n_expert][out][in] buffer -- expert is the outer,
 * slowest-varying axis, and each expert's matrix is row-major [out][in] with
 * the input dim innermost.  A ds4_tensor stores this as dim[0]=in, dim[1]=out,
 * dim[2]=expert; the test fills and indexes slice `e` at e*out*in.  The routing
 * case below forces a top-8 of {248..255}, so an implementation that treats
 * the expert axis as anything else than the outer block cannot pass.
 *
 * Four behaviours are shown to fail by manual ablation in the hook (recorded
 * in the task report): the expert gather (indices[k] -> 0), the top-k
 * renormalisation (raw full-softmax probabilities), the shared sigmoid gate
 * (g = 1) and the silu placement (silu of the product).  The main case uses a
 * shared gate near 0.5, so setting it to 1 changes the output far beyond
 * tolerance.
 *
 * Cases: token A routes to the biased set {248..255}; token B re-runs that set
 * with a non-uniform input; token C routes to a set that overlaps A's but is
 * not identical (a mixed expert gather); token D doubles each shared-expert matrix (see the case comment)
 * three matrices so the shared contribution dominates the routed sum.
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-moe
 *   ./external/ds4/tests/test_qwen35_moe
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"

/* Ornith-1.5-35B-A3B (DS4_SHAPE_ORNITH15) MoE geometry. */
enum {
    N_EMBD        = 2048,
    N_FF_EXP      = 512,
    N_EXPERT      = 256,
    N_EXPERT_USED = 8,
};

/* Every routed expert slice is [out][in] with the input dim innermost, so the
 * expert axis is the slowest-varying (outer) axis of the stack.  gate/up are
 * [n_ff_exp][n_embd]; down is [n_embd][n_ff_exp]. */
#define GATE_EXPERT_FLOATS ((uint64_t)N_FF_EXP * N_EMBD)
#define DOWN_EXPERT_FLOATS ((uint64_t)N_EMBD * N_FF_EXP)
#define MOE_MODEL_FLOATS                                              \
    ((uint64_t)N_EXPERT * N_EMBD +           /* ffn_gate_inp          */ \
     3u * (uint64_t)N_FF_EXP * N_EMBD +      /* shexp gate/up/down    */ \
     (uint64_t)N_EMBD +                      /* ffn_gate_inp_shexp    */ \
     3u * (uint64_t)N_EXPERT * GATE_EXPERT_FLOATS) /* exps gate/up/down */
#define MOE_MODEL_BYTES ((size_t)MOE_MODEL_FLOATS * sizeof(float))

/* Must match the struct ds4.c defines for the hook.  Pointers are explicit; the
 * stacks are [n_expert][out][in] as described above. */
typedef struct {
    const float *ffn_gate_inp;       /* [n_expert][n_embd] router */
    const float *ffn_gate_exps;      /* [n_expert][n_ff_exp][n_embd] */
    const float *ffn_up_exps;        /* [n_expert][n_ff_exp][n_embd] */
    const float *ffn_down_exps;      /* [n_expert][n_embd][n_ff_exp] */
    const float *ffn_gate_inp_shexp; /* [n_embd] */
    const float *ffn_gate_shexp;     /* [n_ff_exp][n_embd] */
    const float *ffn_up_shexp;       /* [n_ff_exp][n_embd] */
    const float *ffn_down_shexp;     /* [n_embd][n_ff_exp] */
} ds4_test_qwen35_moe_weights;

int ds4_test_qwen35_moe_forward(const ds4_test_qwen35_moe_weights *w,
                                const float *x, float *out);

uint32_t ds4_test_qwen35_moe_route(const float *logits, uint32_t n_expert,
                                   uint32_t top_k, uint32_t *indices,
                                   float *weights);
float ds4_test_qwen35_shared_expert_gate(float dot);

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

/* Deterministic, zero-mean-ish synthetic values. */
static float synth(uint32_t o, uint32_t i, float scale) {
    uint32_t v = (o * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
    v ^= v >> 13;
    v *= 2246822519u;
    v ^= v >> 16;
    const int32_t s = (int32_t)((v >> 8) % 2001u) - 1000;
    return scale * (float)s / 1000.0f;
}

static float sigmoid1(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

static float silu1(float x) {
    return x * sigmoid1(x);
}

static float dot1(const float *a, const float *b, uint64_t n) {
    double acc = 0.0;
    for (uint64_t i = 0; i < n; i++) acc += (double)a[i] * b[i];
    return (float)acc;
}

static void matvec1(float *out, const float *weight, uint32_t in_dim,
                    uint32_t out_dim, const float *x) {
    for (uint32_t o = 0; o < out_dim; o++) {
        const float *row = weight + (uint64_t)o * in_dim;
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)row[i] * x[i];
        out[o] = (float)acc;
    }
}

/* Independent scalar reference for the whole MoE FFN.  Routing and the shared
 * gate are consumed from the shipped hooks; the matmuls and the combination are
 * scalar.  The expert slice for `e` is the outer block e*out*in.  The routed and
 * shared partials are written separately so a case can assert which dominates. */
static void reference_split(const ds4_test_qwen35_moe_weights *w,
                            const float *x, float *routed, float *shared) {
    float logits[N_EXPERT];
    uint32_t indices[N_EXPERT_USED];
    float weights[N_EXPERT_USED];
    float gate[N_FF_EXP], up[N_FF_EXP], h[N_FF_EXP];

    for (uint32_t e = 0; e < N_EXPERT; e++)
        logits[e] = dot1(w->ffn_gate_inp + (uint64_t)e * N_EMBD, x, N_EMBD);

    const uint32_t used = ds4_test_qwen35_moe_route(
        logits, N_EXPERT, N_EXPERT_USED, indices, weights);
    require_ok(used == N_EXPERT_USED, "reference router returns top-8");

    for (uint32_t d = 0; d < N_EMBD; d++) {
        routed[d] = 0.0f;
        shared[d] = 0.0f;
    }

    for (uint32_t k = 0; k < used; k++) {
        const uint32_t e = indices[k];
        const float *gate_e = w->ffn_gate_exps + (uint64_t)e * GATE_EXPERT_FLOATS;
        const float *up_e = w->ffn_up_exps + (uint64_t)e * GATE_EXPERT_FLOATS;
        const float *down_e = w->ffn_down_exps + (uint64_t)e * DOWN_EXPERT_FLOATS;

        matvec1(gate, gate_e, N_EMBD, N_FF_EXP, x);
        matvec1(up, up_e, N_EMBD, N_FF_EXP, x);
        for (uint32_t j = 0; j < N_FF_EXP; j++)
            h[j] = silu1(gate[j]) * up[j];

        for (uint32_t d = 0; d < N_EMBD; d++) {
            const float *row = down_e + (uint64_t)d * N_FF_EXP;
            double acc = 0.0;
            for (uint32_t j = 0; j < N_FF_EXP; j++)
                acc += (double)row[j] * h[j];
            routed[d] += weights[k] * (float)acc;
        }
    }

    /* Shared expert: silu(gate_s . x) * (up_s . x), down, times
     * sigmoid(ffn_gate_inp_shexp . x). */
    matvec1(gate, w->ffn_gate_shexp, N_EMBD, N_FF_EXP, x);
    matvec1(up, w->ffn_up_shexp, N_EMBD, N_FF_EXP, x);
    for (uint32_t j = 0; j < N_FF_EXP; j++)
        h[j] = silu1(gate[j]) * up[j];
    {
        const float sgate =
            ds4_test_qwen35_shared_expert_gate(dot1(w->ffn_gate_inp_shexp, x, N_EMBD));
        for (uint32_t d = 0; d < N_EMBD; d++) {
            const float *row = w->ffn_down_shexp + (uint64_t)d * N_FF_EXP;
            double acc = 0.0;
            for (uint32_t j = 0; j < N_FF_EXP; j++)
                acc += (double)row[j] * h[j];
            shared[d] = sgate * (float)acc;
        }
    }
}

static void reference_forward(const ds4_test_qwen35_moe_weights *w,
                              const float *x, float *out) {
    float routed[N_EMBD];
    float shared[N_EMBD];
    reference_split(w, x, routed, shared);
    for (uint32_t d = 0; d < N_EMBD; d++) out[d] = routed[d] + shared[d];
}

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

static float max_abs(const float *a, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i]));
    return m;
}

/* Fill one routed expert's [out][in] gate/up/down slices.  Distinct per-expert
 * seeds mean a wrong expert axis or gather cannot pass. */
static void fill_expert(float *gate_exps, float *up_exps, float *down_exps,
                        uint32_t e) {
    const uint64_t gbase = (uint64_t)e * GATE_EXPERT_FLOATS;
    const uint64_t dbase = (uint64_t)e * DOWN_EXPERT_FLOATS;
    for (uint32_t o = 0; o < N_FF_EXP; o++) {
        for (uint32_t i = 0; i < N_EMBD; i++) {
            gate_exps[gbase + (uint64_t)o * N_EMBD + i] =
                synth(e * 131u + o, i, 0.03f);
            up_exps[gbase + (uint64_t)o * N_EMBD + i] =
                synth(e * 131u + o + 17u, i, 0.03f);
        }
    }
    for (uint32_t d = 0; d < N_EMBD; d++) {
        for (uint32_t j = 0; j < N_FF_EXP; j++) {
            down_exps[dbase + (uint64_t)d * N_FF_EXP + j] =
                synth(e * 131u + d + 53u, j, 0.03f);
        }
    }
}

static void run_hook(const ds4_test_qwen35_moe_weights *w, const float *x,
                     float *out) {
    require_ok(ds4_test_qwen35_moe_forward(w, x, out) == 0,
               "qwen35 moe forward");
}

/* Route a token with the shipped router and copy the top-8 indices out.
 * Returns the number of experts used (always N_EXPERT_USED here). */
static uint32_t route_only(const ds4_test_qwen35_moe_weights *w, const float *x,
                           uint32_t *indices) {
    float logits[N_EXPERT];
    float weights[N_EXPERT_USED];
    for (uint32_t e = 0; e < N_EXPERT; e++)
        logits[e] = dot1(w->ffn_gate_inp + (uint64_t)e * N_EMBD, x, N_EMBD);
    const uint32_t used = ds4_test_qwen35_moe_route(
        logits, N_EXPERT, N_EXPERT_USED, indices, weights);
    require_ok(used == N_EXPERT_USED, "router returns top-8");
    return used;
}

/* One case: route a token, fill exactly the selected experts, then require the
 * hook and the scalar reference to agree.  Returns the top-1 index so the
 * caller can assert the expert-axis wiring. */
static uint32_t run_case(const ds4_test_qwen35_moe_weights *w,
                         float *gate_exps, float *up_exps, float *down_exps,
                         const float *x, const char *what, float tolerance) {
    uint32_t indices[N_EXPERT_USED];
    const uint32_t used = route_only(w, x, indices);

    for (uint32_t k = 0; k < used; k++)
        fill_expert(gate_exps, up_exps, down_exps, indices[k]);

    float expect[N_EMBD];
    float actual[N_EMBD];
    reference_forward(w, x, expect);
    run_hook(w, x, actual);
    for (uint32_t d = 0; d < N_EMBD; d++)
        require_close(what, actual[d], expect[d], tolerance);

    /* The hook and the reference must agree on routing; a disagreement here
     * means the FFN is being driven by different experts than the reference. */
    require_ok(max_abs_diff(actual, expect, N_EMBD) <= tolerance,
               "moe output matches the reference");
    return indices[0];
}

int main(void) {
    uint8_t *model = mmap(NULL, MOE_MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    float *gate_inp = (float *)model;
    float *gate_shexp = gate_inp + (uint64_t)N_EXPERT * N_EMBD;
    float *up_shexp = gate_shexp + (uint64_t)N_FF_EXP * N_EMBD;
    float *down_shexp = up_shexp + (uint64_t)N_FF_EXP * N_EMBD;
    float *gate_inp_shexp = down_shexp + (uint64_t)N_EMBD * N_FF_EXP;
    float *gate_exps = gate_inp_shexp + N_EMBD;
    float *up_exps = gate_exps + (uint64_t)N_EXPERT * GATE_EXPERT_FLOATS;
    float *down_exps = up_exps + (uint64_t)N_EXPERT * GATE_EXPERT_FLOATS;
    require_ok((uint8_t *)(down_exps + (uint64_t)N_EXPERT * DOWN_EXPERT_FLOATS) <=
                   model + MOE_MODEL_BYTES,
               "synthetic weights fit the mapping");

    /* Router rows: a small random projection plus a strong fixed bias on
     * experts 248..255.  With the all-ones token the logit is the row sum, so
     * the biased experts win the top-8 outright; if the stack were indexed on
     * the wrong axis (or the gather replaced with expert 0) the outputs differ.
     * The bias is small enough that the top-8 still carry only ~1/8 of the full
     * softmax mass, so dropping the renormalisation changes the result. */
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        for (uint32_t i = 0; i < N_EMBD; i++)
            gate_inp[(uint64_t)e * N_EMBD + i] = synth(e, i, 0.005f);
        if (e >= 248u) gate_inp[(uint64_t)e * N_EMBD] += 1.5f;
    }

    for (uint32_t o = 0; o < N_FF_EXP; o++) {
        for (uint32_t i = 0; i < N_EMBD; i++) {
            gate_shexp[(uint64_t)o * N_EMBD + i] = synth(o + 401u, i, 0.02f);
            up_shexp[(uint64_t)o * N_EMBD + i] = synth(o + 503u, i, 0.02f);
        }
    }
    for (uint32_t d = 0; d < N_EMBD; d++) {
        for (uint32_t j = 0; j < N_FF_EXP; j++)
            down_shexp[(uint64_t)d * N_FF_EXP + j] = synth(d + 601u, j, 0.02f);
    }
    for (uint32_t d = 0; d < N_EMBD; d++)
        gate_inp_shexp[d] = synth(d, 701u, 0.001f);

    ds4_test_qwen35_moe_weights w = {
        .ffn_gate_inp = gate_inp,
        .ffn_gate_exps = gate_exps,
        .ffn_up_exps = up_exps,
        .ffn_down_exps = down_exps,
        .ffn_gate_inp_shexp = gate_inp_shexp,
        .ffn_gate_shexp = gate_shexp,
        .ffn_up_shexp = up_shexp,
        .ffn_down_shexp = down_shexp,
    };

    /* Token A: all ones.  The 248..255 bias makes those the routed top-8, which
     * is deliberately not 0..7. */
    float xa[N_EMBD];
    for (uint32_t d = 0; d < N_EMBD; d++) xa[d] = 1.0f;

    const uint32_t top = run_case(&w, gate_exps, up_exps, down_exps, xa,
                                  "qwen35 moe token A (biased top-8)", 1e-5f);
    require_ok(top >= 8u, "token A routed top-8 is not 0..7");
    require_ok(top >= 248u, "token A routed top-1 is one of the biased experts");

    /* The shared gate must be meaningfully below 1, or the g=1 ablation is
     * vacuous. */
    {
        const float g = ds4_test_qwen35_shared_expert_gate(
            dot1(gate_inp_shexp, xa, N_EMBD));
        require_ok(g > 0.05f && g < 0.95f, "shared gate is away from 1");
    }

    /* Token B: a varied token.  With this router bias the same biased top-8
     * wins (in a different order), so token B re-runs the gather on the same
     * expert set with a non-uniform input; token C below supplies the
     * overlapping-but-different set. */
    float xb[N_EMBD];
    for (uint32_t d = 0; d < N_EMBD; d++) xb[d] = synth(d, 3u, 0.5f);
    run_case(&w, gate_exps, up_exps, down_exps, xb,
             "qwen35 moe token B (varied token)", 1e-5f);

    /* Token C: all ones except the biased router dimension, weakened to 0.15.
     * That lets unbiased experts into the top-8, so token C routes to a set
     * overlapping token A's {248..255} but not identical -- the gather must
     * handle a mixed expert set.  Assert the overlap is partial. */
    float xc[N_EMBD];
    for (uint32_t d = 0; d < N_EMBD; d++) xc[d] = 1.0f;
    xc[0] = 0.15f;
    {
        uint32_t ia[N_EXPERT_USED];
        uint32_t ic[N_EXPERT_USED];
        route_only(&w, xa, ia);
        route_only(&w, xc, ic);
        uint32_t overlap = 0;
        for (uint32_t k = 0; k < N_EXPERT_USED; k++)
            for (uint32_t j = 0; j < N_EXPERT_USED; j++)
                if (ia[k] == ic[j]) overlap++;
        require_ok(overlap >= 1u && overlap < N_EXPERT_USED,
                   "token C routed set overlaps but differs from token A");
    }
    run_case(&w, gate_exps, up_exps, down_exps, xc,
             "qwen35 moe token C (overlapping expert set)", 1e-5f);

    /* Token D: shared-expert-dominant.  The routed experts keep their nonzero
     * synthetic weights, but each of the shared expert's three matrices is
     * doubled (compounding to ~8x before the SiLU; measured ~3.6x on the output),
     * so the shared term dominates the routed sum.  Assert dominance on the
     * scalar reference's own partials, then compare the hook against it. */
    float xd[N_EMBD];
    for (uint32_t d = 0; d < N_EMBD; d++) xd[d] = synth(d, 5u, 0.5f);

    const uint64_t shared_floats = (uint64_t)N_FF_EXP * N_EMBD;
    for (uint64_t i = 0; i < shared_floats; i++) gate_shexp[i] *= 2.0f;
    for (uint64_t i = 0; i < shared_floats; i++) up_shexp[i] *= 2.0f;
    for (uint64_t i = 0; i < shared_floats; i++) down_shexp[i] *= 2.0f;
    {
        uint32_t indices[N_EXPERT_USED];
        float routed[N_EMBD];
        float shared[N_EMBD];
        route_only(&w, xd, indices);
        for (uint32_t k = 0; k < N_EXPERT_USED; k++)
            fill_expert(gate_exps, up_exps, down_exps, indices[k]);
        reference_split(&w, xd, routed, shared);
        require_ok(max_abs(shared, N_EMBD) > max_abs(routed, N_EMBD),
                   "token D shared contribution dominates the routed sum");
    }
    run_case(&w, gate_exps, up_exps, down_exps, xd,
             "qwen35 moe token D (shared-dominant)", 1e-5f);
    for (uint64_t i = 0; i < shared_floats; i++) gate_shexp[i] *= 0.5f;
    for (uint64_t i = 0; i < shared_floats; i++) up_shexp[i] *= 0.5f;
    for (uint64_t i = 0; i < shared_floats; i++) down_shexp[i] *= 0.5f;

    /* The hook must reject a null weight pointer and a null row. */
    {
        float out[N_EMBD];
        require_ok(ds4_test_qwen35_moe_forward(NULL, xa, out) != 0,
                   "moe forward rejects a null weight struct");
        require_ok(ds4_test_qwen35_moe_forward(&w, NULL, out) != 0,
                   "moe forward rejects a null input");
    }

    munmap(model, MOE_MODEL_BYTES);
    puts("qwen35 MoE feed-forward layer core: PASS");
    return 0;
}
