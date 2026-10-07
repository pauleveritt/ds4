// qwen35 verify and the DFlash drafter: few-row Q4_K, Q6_K and Q4_0
// matvecs on 8x8 simdgroup matrices.
//
// Ported from llama.cpp's ggml/src/ggml-metal/kernels/mul_mv_mma.metal
// (ggml-org/llama.cpp #29869, at a46709b; MIT, the ggml authors):
// mul_mv_mma_tile_init, mul_mv_mma_src0_row, mul_mv_mma_src1_row,
// mul_mv_mma_store and kernel_mul_mv_mma_gen at one 8-row src1 tile (RT 1),
// with NT 1, 2 or 4 weight tiles, on moe.metal's dequantize_q4_K with
// word loads and a port of llama.cpp's dequantize_q6_K (both below); and
// for Q4_0, mul_mv_mma_1024_plus, mul_mv_mma_q4_0, load_mma_blk_a, load_mma_blk_b and
// kernel_mul_mv_mma_blk at RT 1 (the row length a kernel argument, not a
// function constant).
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

// moe.metal's dequantize_q4_K from four 32-bit loads instead of sixteen
// byte loads: the same 16 weights, each dl*q - ml on the same dl, ml and q,
// so the same floats in the same order. A piece's 16 quant bytes start at
// qs + 32*(il/4) + 16*(il&1), 4-byte aligned in a 144-byte block whose qs
// starts at byte 16.
template <typename type4x4>
void qwen35_mma_dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uint32_t * q = (device const uint32_t *)(xb->qs + (il/4)*32 + 16*(il&1));

    const short is = (il/4)*2;
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ?
        (float)xb->d :
        (float)xb->d * (1.0f / 16.0f);
    const float min = (float)xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const uint32_t mask = il < 2 ? 0x0F0F0F0F : 0xF0F0F0F0;
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const uint32_t w = q[i] & mask;
        reg[i][0] = dl * (float)( w        & 0xFF) - ml;
        reg[i][1] = dl * (float)((w >>  8) & 0xFF) - ml;
        reg[i][2] = dl * (float)((w >> 16) & 0xFF) - ml;
        reg[i][3] = dl * (float)( w >> 24        ) - ml;
    }
}

// the same as qwen35_mma_dequantize_q4_K from one 16-byte load, the
// default; a piece's bytes are 16-byte aligned when the block is, and the
// host takes the 32-bit loads when the weights are not. At T = 8 the
// per-byte loads cost 253.6 us at 5120->17408, 32-bit loads 236.2 and one
// 16-byte load 209.0 (docs/superpowers/research/2026-10-07-mma-q4k-word.md
// in the ds4-engine host repository).
template <typename type4x4>
void qwen35_mma_dequantize_q4_K_vec(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    const uint4 qv = *(device const uint4 *)(xb->qs + (il/4)*32 + 16*(il&1));

    const short is = (il/4)*2;
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ?
        (float)xb->d :
        (float)xb->d * (1.0f / 16.0f);
    const float min = (float)xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const uint32_t mask = il < 2 ? 0x0F0F0F0F : 0xF0F0F0F0;
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const uint32_t w = qv[i] & mask;
        reg[i][0] = dl * (float)( w        & 0xFF) - ml;
        reg[i][1] = dl * (float)((w >>  8) & 0xFF) - ml;
        reg[i][2] = dl * (float)((w >> 16) & 0xFF) - ml;
        reg[i][3] = dl * (float)( w >> 24        ) - ml;
    }
}

// where a lane's 16-weight piece il sits in a Q4_K block: its quant bytes,
// its sub-block j's scale and min bytes within the 12 scale bytes read as
// three words, and its nibble
struct qwen35_q4_K_lane {
    ushort qs;
    uint   shift;
    bool   upper;
    bool   high;
};

inline qwen35_q4_K_lane qwen35_q4_K_lane_init(short il) {
    qwen35_q4_K_lane ln;
    const short j = 2*(il/4) + (il & 3)/2;
    ln.qs    = 16 + 32*(il/4) + 16*(il & 1);
    ln.shift = 8*(j & 3);
    ln.upper = j >= 4;
    ln.high  = (il & 3) >= 2;
    return ln;
}

// qwen35_mma_dequantize_q4_K_vec with d, dmin and the 12 scale bytes from
// one 16-byte load of the block's head, the 6-bit scale and min cut from its
// words (get_scale_min_k4_just2's bytes j, j + 4 and j + 8 or j - 4 are bytes
// j%4 of words 1, 2 and 3): the same integers, so the same dl, ml and floats
inline void qwen35_q4_K_lane_dequantize(device const block_q4_K * xb, thread const qwen35_q4_K_lane & ln,
                                        thread float4x4 & reg) {
    const uint4 h  = *(device const uint4 *) xb;
    const uint4 qv = *(device const uint4 *)((device const uchar *) xb + ln.qs);

    const uint s0 = h.y >> ln.shift;
    const uint s1 = h.z >> ln.shift;
    const uint s2 = h.w >> ln.shift;
    const uint sc = ln.upper ? ((s2 & 0xF)        | ((s0 & 0xC0) >> 2)) : (s0 & 63);
    const uint mn = ln.upper ? (((s2 >> 4) & 0xF) | ((s1 & 0xC0) >> 2)) : (s1 & 63);

    const half2 dm = as_type<half2>(h.x);
    const float d = ln.high ?
        (float)dm[0] * (1.0f / 16.0f) :
        (float)dm[0];
    const float min = (float)dm[1];
    const float dl = d * (float)sc;
    const float ml = min * (float)mn;

    const uint32_t mask = ln.high ? 0xF0F0F0F0 : 0x0F0F0F0F;
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const uint32_t w = qv[i] & mask;
        reg[i][0] = dl * (float)( w        & 0xFF) - ml;
        reg[i][1] = dl * (float)((w >>  8) & 0xFF) - ml;
        reg[i][2] = dl * (float)((w >> 16) & 0xFF) - ml;
        reg[i][3] = dl * (float)( w >> 24        ) - ml;
    }
}

// the head-load dequantizer with the piece's place worked out per call
template <typename type4x4>
void qwen35_mma_dequantize_q4_K_hdr(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    const qwen35_q4_K_lane ln = qwen35_q4_K_lane_init(il);
    qwen35_q4_K_lane_dequantize(xb, ln, reg);
}

// kernel_qwen35_mma_f32 for Q4_K on the head-load dequantizer with the
// piece's place worked out once per lane: simdgroup sgitg takes the chunks
// g = sgitg + k*NSG, so when NSG is a multiple of 4 (always, for in_dim a
// multiple of 256) every chunk has g%4 = sgitg%4 and a lane's piece il =
// 4*(g%4) + fn/2 is the same in every block
template<short NT>
kernel void kernel_qwen35_mma_q4_K_lane_f32(
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

    device const block_q4_K * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q4_K *) qwen35_mma_src0_row(tile, args, src0, t);
    }

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
    const qwen35_q4_K_lane lane = qwen35_q4_K_lane_init(4*(sgitg & 3) + tile.fn/2);

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

        const qwen35_q4_K_lane ln = NSG % 4 == 0 ? lane : qwen35_q4_K_lane_init(4*(g & 3) + tile.fn/2);

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            qwen35_q4_K_lane_dequantize(x[t] + g/4, ln, w);

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

// nl 16: a 256-weight super-block holds sixteen 16-weight pieces
typedef decltype(kernel_qwen35_mma_f32<1, block_q4_K, 16, dequantize_q4_K>) qwen35_mma_t;

template [[host_name("kernel_qwen35_mma_q4_K_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q4_K, 16, dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q4_K, 16, dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q4_K, 16, dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_word_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q4_K, 16, qwen35_mma_dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_word_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q4_K, 16, qwen35_mma_dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_word_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q4_K, 16, qwen35_mma_dequantize_q4_K>;
template [[host_name("kernel_qwen35_mma_q4_K_vec_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q4_K, 16, qwen35_mma_dequantize_q4_K_vec>;
template [[host_name("kernel_qwen35_mma_q4_K_vec_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q4_K, 16, qwen35_mma_dequantize_q4_K_vec>;
template [[host_name("kernel_qwen35_mma_q4_K_vec_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q4_K, 16, qwen35_mma_dequantize_q4_K_vec>;
template [[host_name("kernel_qwen35_mma_q4_K_hdr_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q4_K, 16, qwen35_mma_dequantize_q4_K_hdr>;
template [[host_name("kernel_qwen35_mma_q4_K_hdr_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q4_K, 16, qwen35_mma_dequantize_q4_K_hdr>;
template [[host_name("kernel_qwen35_mma_q4_K_hdr_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q4_K, 16, qwen35_mma_dequantize_q4_K_hdr>;
template [[host_name("kernel_qwen35_mma_q4_K_lane_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_q4_K_lane_f32<1>;
template [[host_name("kernel_qwen35_mma_q4_K_lane_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_q4_K_lane_f32<2>;
template [[host_name("kernel_qwen35_mma_q4_K_lane_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_q4_K_lane_f32<4>;
template [[host_name("kernel_qwen35_mma_q6_K_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<1, block_q6_K, 16, qwen35_mma_dequantize_q6_K>;
template [[host_name("kernel_qwen35_mma_q6_K_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<2, block_q6_K, 16, qwen35_mma_dequantize_q6_K>;
template [[host_name("kernel_qwen35_mma_q6_K_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_f32<4, block_q6_K, 16, qwen35_mma_dequantize_q6_K>;

constexpr constant static ushort qwen35_mma_f16_1024_bits = 0x6400;
constexpr constant static half   qwen35_mma_f16_1024      = 1024.0h;

// the halves 1024 + q for integers q < 1024: exact normal values, unlike the subnormal q*2^-24 that Metal may flush to zero
inline half2 qwen35_mma_1024_plus(ushort2 q) {
    return as_type<half2>(q | qwen35_mma_f16_1024_bits);
}

// llama.cpp's mul_mv_mma_q4_0: a K step is one 32-weight block, its src1 values split into halves b0 and b1;
// MMA step s uses half b1 when s is odd and the .y value of a pair when s >= 2.
// A lane (m, j) holds qs ushorts j and j + 1 (j even); a high nibble stays in place as 16*q, so b1 is divided by 16.
// B lane k = fm holds src1 values 2*k, 2*k + 1 (b0, low nibbles) and 2*k + 16, 2*k + 17 (b1, high nibbles) of a block
struct qwen35_mma_q4_0 {
    typedef ds4_dense_block_q4_0 block;
    typedef ushort2              quants;

    // weights per block, and the float2 offset of b1 in a src1 block
    enum { qk = 32, b1 = 32/4 };

    static short a_off(short fn) { return 1 + fn; }
    static short b_off(short fm) { return 2*fm; }

    static quants load(device const ushort * qs) { return ushort2(qs[0], qs[1]); }
    static float2 prep_b1(float2 v) {
        constexpr float hi_scale = 1.0f/16;
        return v*hi_scale;
    }

    static bool b1_step(short s) { return s % 2 != 0; }
    static bool y_step (short s) { return s >= 2; }

    static half2 frag(quants q, short s) {
        constexpr ushort lo_mask = 0x000F;
        constexpr ushort hi_mask = 0x00F0;
        constexpr half   lo_zero = 8.0h;
        constexpr half   hi_zero = 16*lo_zero;

        const ushort2 qq = s < 2 ? q : q >> 8;
        return s % 2 == 0 ? qwen35_mma_1024_plus(qq & lo_mask) - (qwen35_mma_f16_1024 + lo_zero)
                          : qwen35_mma_1024_plus(qq & hi_mask) - (qwen35_mma_f16_1024 + hi_zero);
    }
};

template<typename Q, short NT>
inline void qwen35_mma_load_blk_a(device const ushort * const x[NT], int off, short fn, thread typename Q::quants * q, thread float * d) {
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const ushort * qs = x[t] + off;
        q[t] = Q::load(qs);
        d[t] = as_type<half>(*(qs - Q::a_off(fn)));
    }
}

// src1 columns past ne11 read as zero
template<typename Q>
inline void qwen35_mma_load_blk_b(device const float2 * const y[2], const bool live[2], int ib, thread float2 * b0, thread float2 * b1) {
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        b0[e] = live[e] ? y[e][ib*(Q::qk/2)]         : float2(0.0f);
        b1[e] = live[e] ? y[e][ib*(Q::qk/2) + Q::b1] : float2(0.0f);
    }
}

// few-row mat-mat for 32-weight block types: a threadgroup reads each weight once for 8*NT src0 rows x 8 src1 rows,
// and its NSG simdgroups split K by blocks, the next block's loads issued before the current block's MMAs
template<short NT, typename Q>
kernel void kernel_qwen35_mma_blk_f32(
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

    device const ushort * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const ushort *) qwen35_mma_src0_row(tile, args, src0, t) + Q::a_off(tile.fn);
    }

    device const float2 * y[2];
    bool live[2];
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        live[e] = tile.fn + e < args.ne11;
        y[e] = (device const float2 *) (qwen35_mma_src1_row(tile, args, src1, e) + Q::b_off(tile.fm));
    }

    float acc[NT][2] = {};

    const int nb = args.ne00/Q::qk;

    // a block is d, then the quants
    constexpr short us_blk = sizeof(typename Q::block)/2;

    typename Q::quants q[NT];
    float  d[NT];
    float2 b0[2];
    float2 b1[2];

    const int ib0 = min((int) sgitg, nb - 1);
    qwen35_mma_load_blk_a<Q, NT>(x, ib0*us_blk, tile.fn, q, d);
    qwen35_mma_load_blk_b<Q>(y, live, ib0, b0, b1);

    for (int ib = sgitg; ib < nb; ib += NSG) {
        typename Q::quants qc[NT];
        float  dc[NT];
        float2 b0c[2];
        float2 b1c[2];

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            qc[t] = q[t];
            dc[t] = d[t];
        }
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            b0c[e] = b0[e];
            b1c[e] = Q::prep_b1(b1[e]);
        }

        const int ibn = min(ib + NSG, nb - 1);
        qwen35_mma_load_blk_a<Q, NT>(x, ibn*us_blk, tile.fn, q, d);
        qwen35_mma_load_blk_b<Q>(y, live, ibn, b0, b1);

        simdgroup_float8x8 mp[NT];
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mp[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }

        FOR_UNROLL (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb;
            {
                const float2 v0 = Q::b1_step(s) ? b1c[0] : b0c[0];
                const float2 v1 = Q::b1_step(s) ? b1c[1] : b0c[1];
                mb.thread_elements()[0] = Q::y_step(s) ? v0.y : v0.x;
                mb.thread_elements()[1] = Q::y_step(s) ? v1.y : v1.x;
            }

            FOR_UNROLL (short t = 0; t < NT; ++t) {
                const half2 h = Q::frag(qc[t], s);

                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;

                simdgroup_multiply_accumulate(mp[t], ma, mb, mp[t]);
            }
        }

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            acc[t][0] = fma(dc[t], mp[t].thread_elements()[0], acc[t][0]);
            acc[t][1] = fma(dc[t], mp[t].thread_elements()[1], acc[t][1]);
        }
    }

    qwen35_mma_store<NT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

template [[host_name("kernel_qwen35_mma_q4_0_f32_nt1")]] kernel qwen35_mma_t kernel_qwen35_mma_blk_f32<1, qwen35_mma_q4_0>;
template [[host_name("kernel_qwen35_mma_q4_0_f32_nt2")]] kernel qwen35_mma_t kernel_qwen35_mma_blk_f32<2, qwen35_mma_q4_0>;
template [[host_name("kernel_qwen35_mma_q4_0_f32_nt4")]] kernel qwen35_mma_t kernel_qwen35_mma_blk_f32<4, qwen35_mma_q4_0>;
