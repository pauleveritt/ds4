// qwen35 verify: few-row Q4_K and Q6_K matvecs on 8x8 simdgroup matrices.
//
// Ported from llama.cpp's ggml/src/ggml-metal/kernels/mul_mv_mma.metal
// (ggml-org/llama.cpp #29869, at a46709b; MIT, the ggml authors):
// mul_mv_mma_tile_init, mul_mv_mma_src0_row, mul_mv_mma_src1_row,
// mul_mv_mma_store and kernel_mul_mv_mma_gen at one 8-row src1 tile (RT 1),
// with NT 1, 2 or 4 weight tiles, on moe.metal's dequantize_q4_K and a
// port of llama.cpp's dequantize_q6_K (below).
//
// Changes from llama.cpp: no batch dimensions and no fused residual add;
// src1 rows past ne11 are zero instead of copies of the last row. The tile
// is 8 rows at every row count 1..8 and the host picks NSG (the K split)
// and NT from the weight shape alone (llama.cpp's tiling at RT 1), so
// column t sees the same chunks, the same MMA sequence and the same
// cross-simdgroup reduction whatever ne11 is and whatever the other columns
// hold: output row t does not depend on the row count or on the other rows.
// The sums are not in the one-row kernels' order and differ from them in
// the last bits.

struct ds4_metal_args_qwen35_mma {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne11;
    int32_t  pad0;
    uint64_t nb01;
    uint64_t nb11;
};

// a lane of a few-row MMA tile: A fragment row fm, B fragment columns fn and fn + 1
struct qwen35_mma_tile {
    short fm;
    short fn;
    int   i01;
};

// the A fragment row and the first B fragment column that lane l holds in an 8x8 simdgroup matrix
inline short qwen35_mma_lane_fm(ushort l) { return ((l/4) & 4) + ((l/2) % 4); }
inline short qwen35_mma_lane_fn(ushort l) { return ((l/4) & 2)*2 + (l % 2)*2; }

template<short NT>
inline qwen35_mma_tile qwen35_mma_tile_init(uint3 tgpig, ushort tiisg) {
    qwen35_mma_tile tile;
    tile.fm  = qwen35_mma_lane_fm(tiisg);
    tile.fn  = qwen35_mma_lane_fn(tiisg);
    tile.i01 = tgpig.x*(8*NT);
    return tile;
}

// the src0 row of A fragment row fm in 8-row tile t, clamped to the last row
inline device const char * qwen35_mma_src0_row(
        thread const qwen35_mma_tile & tile, constant ds4_metal_args_qwen35_mma & args, device const char * src0, short t) {
    const int r = min(tile.i01 + 8*t + tile.fm, args.ne01 - 1);
    return src0 + (uint64_t) r*args.nb01;
}

// the src1 row of B fragment column fn + e, clamped to the last row (its values are then replaced by zero)
inline device const float * qwen35_mma_src1_row(
        thread const qwen35_mma_tile & tile, constant ds4_metal_args_qwen35_mma & args, device const char * src1, short e) {
    const int r = min(tile.fn + e, args.ne11 - 1);
    return (device const float *) (src1 + (uint64_t) r*args.nb11);
}

// adds up the K slices of the NSG simdgroups for an 8*NT x 8 output tile and writes its live rows
template<short NT>
inline void qwen35_mma_store(
        thread float (&acc)[NT][2],
        constant ds4_metal_args_qwen35_mma & args,
        device char * dst,
        threadgroup char * shmem,
        thread const qwen35_mma_tile & tile, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * red = (threadgroup float *) shmem;

    FOR_UNROLL (short t = 0; t < NT; ++t) {
        red[(sgitg*NT + t)*64 + 2*tiisg + 0] = acc[t][0];
        red[(sgitg*NT + t)*64 + 2*tiisg + 1] = acc[t][1];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst_f32 = (device float *) dst;

    for (short idx = sgitg*32 + tiisg; idx < NT*64; idx += NSG*32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg*(NT*64) + idx];
        }

        const short t = idx/64;
        const short l = (idx % 64)/2;
        const short e = idx % 2;

        const int r0 = tile.i01 + 8*t + qwen35_mma_lane_fm(l);
        const int r1 = qwen35_mma_lane_fn(l) + e;

        if (r0 < args.ne01 && r1 < args.ne11) {
            dst_f32[(uint64_t) r1*args.ne01 + r0] = sum;
        }
    }
}

// few-row mat-mat for a 16-weight dequantizer: a lane dequantizes 16 consecutive weights of a 64-weight chunk once for all 8 src1 rows.
// MMA step s at MMA-k index j reads chunk weight 16*(j/2) + 8*(j%2) + s.
template<short NT, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_qwen35_mma_f32(
        constant ds4_metal_args_qwen35_mma & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;

    const qwen35_mma_tile tile = qwen35_mma_tile_init<NT>(tgpig, tiisg);

    device const block_q * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q *) qwen35_mma_src0_row(tile, args, src0, t);
    }

    // B lane j = fm reads chunk values 16*(fm/2) + 8*(fm%2) .. +7 of src1 rows fn and fn + 1
    device const float4 * y[2];
    bool live[2];
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        live[e] = tile.fn + e < args.ne11;
        y[e] = (device const float4 *) qwen35_mma_src1_row(tile, args, src1, e) + 4*(tile.fm/2) + 2*(tile.fm%2);
    }

    simdgroup_float8x8 mc[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        mc[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const int nch = args.ne00/64;

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[8];
        {
            const float4 a0 = live[0] ? y[0][16*g + 0] : float4(0.0f);
            const float4 a1 = live[0] ? y[0][16*g + 1] : float4(0.0f);
            const float4 b0 = live[1] ? y[1][16*g + 0] : float4(0.0f);
            const float4 b1 = live[1] ? y[1][16*g + 1] : float4(0.0f);
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                mb[s    ].thread_elements()[0] = a0[s];
                mb[s    ].thread_elements()[1] = b0[s];
                mb[s + 4].thread_elements()[0] = a1[s];
                mb[s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4*g + tile.fn/2;

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            dequantize_func(x[t] + ci/nl, ci%nl, w);

            FOR_UNROLL (short s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s/4    ][s%4];
                ma.thread_elements()[1] = w[s/4 + 2][s%4];

                simdgroup_multiply_accumulate(mc[t], ma, mb[s], mc[t]);
            }
        }
    }

    float acc[NT][2];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        acc[t][0] = mc[t].thread_elements()[0];
        acc[t][1] = mc[t].thread_elements()[1];
    }

    qwen35_mma_store<NT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

// llama.cpp's dequantize_q6_K (ggml-metal/kernels/dequantize.h, a46709b):
// the same 16 weights as moe.metal's dequantize_q6_K, from 16-bit loads and
// whole-word masks, as d*sc*q - 32*d*sc instead of d*sc*(q - 32). moe.metal's
// per-byte version made this kernel about 5x slower than the one-row Q6_K
// matvec at one row.
template <typename type4x4>
void qwen35_mma_dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// nl 16: a 256-weight super-block holds sixteen 16-weight pieces
typedef decltype(kernel_qwen35_mma_f32<1, block_q4_K, 16, dequantize_q4_K>) qwen35_mma_t;

template [[host_name("kernel_qwen35_mma_q4_K_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q4_K, 16, dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q4_K, 16, dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q4_K, 16, dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q6_K_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q6_K, 16, qwen35_mma_dequantize_q6_K>;
template [[host_name("kernel_qwen35_mma_q6_K_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q6_K, 16, qwen35_mma_dequantize_q6_K>;
template [[host_name("kernel_qwen35_mma_q6_K_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q6_K, 16, qwen35_mma_dequantize_q6_K>;
