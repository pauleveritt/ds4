#!/usr/bin/env python3
"""Apply the dot4 vectorization to metal/moe.metal (idempotent check inside)."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else 'metal/moe.metal'
s = open(p).read()
if 'ds4_mellum_q8_0_pair_dot4' in s:
    print('already applied'); sys.exit(0)
helper = '''
/*
 * Shared Q8_0 row dot for the Mellum MoE kernels.  Four elements per step so
 * the block scale is converted once per four values instead of once per value.
 * Every Mellum MoE path uses it, so batch-equals-decode bitwise equality holds.
 *
 * NOTE: this changes intra-row summation order versus the strided loop it
 * replaces, which costs accuracy against the llama.cpp reference (logit RMS
 * 0.00179 -> 0.00911) while gaining ~45% prefill and ~26% decode.  See the
 * research log; adopting it is a deliberate trade, not a free win.
 */
template <typename ROW>
static inline void ds4_mellum_q8_0_pair_dot4(ROW gate_row, ROW up_row,
                                             device const float *x,
                                             uint in_dim, uint tid, uint ntg,
                                             thread float &acc_gate,
                                             thread float &acc_up) {
    for (uint kb = tid * 4u; kb < in_dim; kb += ntg * 4u) {
        const uint block = kb >> 5, el = kb & 31u;
        const float gd = (float)gate_row[block].d;
        const float ud = (float)up_row[block].d;
        for (uint j = 0; j < 4u; j++) {
            const float xv = x[kb + j];
            acc_gate += gd * (float)gate_row[block].qs[el + j] * xv;
            acc_up += ud * (float)up_row[block].qs[el + j] * xv;
        }
    }
}

template <typename ROW>
static inline float ds4_mellum_q8_0_row_dot4(ROW row, device const float *v,
                                             uint dim, uint tid, uint ntg) {
    float acc = 0.0f;
    for (uint kb = tid * 4u; kb < dim; kb += ntg * 4u) {
        const uint block = kb >> 5, el = kb & 31u;
        const float d = (float)row[block].d;
        for (uint j = 0; j < 4u; j++) {
            acc += d * (float)row[block].qs[el + j] * v[kb + j];
        }
    }
    return acc;
}

'''
anchor = "/* The pinned Mellum GGUF is an all-Q8_0 numerical oracle."
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)
reps = [
("""    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint k = tid; k < args.in_dim; k += ntg) {
        const uint block = k / QK8_0;
        const uint element = k - block * QK8_0;
        const float xv = x[k];
        acc_gate += (float)gate_row[block].d *
                    (float)gate_row[block].qs[element] * xv;
        acc_up += (float)up_row[block].d *
                  (float)up_row[block].qs[element] * xv;
    }""",
"""    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    ds4_mellum_q8_0_pair_dot4(gate_row, up_row, x, args.in_dim, tid, ntg,
                              acc_gate, acc_up);"""),
("""    device const float *token_x = x + (uint64_t)token * args.in_dim;
    float acc_gate = 0.0f, acc_up = 0.0f;
    for (uint k = tid; k < args.in_dim; k += ntg) {
        const uint block = k / QK8_0, element = k - block * QK8_0;
        const float xv = token_x[k];
        acc_gate += (float)gate_row[block].d * (float)gate_row[block].qs[element] * xv;
        acc_up += (float)up_row[block].d * (float)up_row[block].qs[element] * xv;
    }""",
"""    device const float *token_x = x + (uint64_t)token * args.in_dim;
    float acc_gate = 0.0f, acc_up = 0.0f;
    ds4_mellum_q8_0_pair_dot4(gate_row, up_row, token_x, args.in_dim, tid, ntg,
                              acc_gate, acc_up);"""),
("""        device const float *token_x = x + (uint64_t)token * args.in_dim;
        float acc_gate = 0.0f, acc_up = 0.0f;
        for (uint k = tid; k < args.in_dim; k += ntg) {
            const uint block = k / QK8_0, element = k - block * QK8_0;
            const float xv = token_x[k];
            acc_gate += (float)gate_row[block].d *
                (float)gate_row[block].qs[element] * xv;
            acc_up += (float)up_row[block].d *
                (float)up_row[block].qs[element] * xv;
        }""",
"""        device const float *token_x = x + (uint64_t)token * args.in_dim;
        float acc_gate = 0.0f, acc_up = 0.0f;
        ds4_mellum_q8_0_pair_dot4(gate_row, up_row, token_x, args.in_dim, tid,
                                  ntg, acc_gate, acc_up);"""),
("""            device const float *slot_mid = mid + (uint64_t)slot * args.mid_dim;
            for (uint k = tid; k < args.mid_dim; k += ntg) {
                const uint block = k / QK8_0;
                const uint element = k - block * QK8_0;
                acc += (float)down_row[block].d *
                       (float)down_row[block].qs[element] * slot_mid[k];
            }""",
"""            device const float *slot_mid = mid + (uint64_t)slot * args.mid_dim;
            acc = ds4_mellum_q8_0_row_dot4(down_row, slot_mid, args.mid_dim,
                                           tid, ntg);"""),
("""            device const float *slot_mid =
                mid + mid_base + (uint64_t)slot * args.mid_dim;
            for (uint k = tid; k < args.mid_dim; k += ntg) {
                const uint block = k / QK8_0, element = k - block * QK8_0;
                acc += (float)down_row[block].d *
                    (float)down_row[block].qs[element] * slot_mid[k];
            }""",
"""            device const float *slot_mid =
                mid + mid_base + (uint64_t)slot * args.mid_dim;
            acc = ds4_mellum_q8_0_row_dot4(down_row, slot_mid, args.mid_dim,
                                           tid, ntg);"""),
]
for o, n in reps:
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n, 1)
open(p, 'w').write(s)
print('applied')
