/* DFlash2 drafter kernels for dense qwen35 (the drafter graph as llama.cpp
 * runs it, src/models/dflash.cpp at a46709b): the per-head RMS norm with
 * NeoX rope, the two-tap block convolution, non-causal windowed attention
 * over the K/V ring and the block's own rows, and the selector's lattice
 * scores.  Transients are f32; the ring holds f16 K and V. */

/* --- target feature capture ----------------------------------------------- */

struct ds4_metal_args_dflash_capture {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;        /* taps per feature row */
    uint32_t aux_index;    /* this tap's slot in the row */
    uint32_t src_row0;
    uint32_t dst_row0;
    uint32_t n_slots;      /* feature rows in the buffer; row dst_row0 + r goes to (dst_row0 + r) % n_slots */
    uint32_t pad0;
};

/* antirez's kernel_dflash_capture_rows (laguna-s2.1, 8f620f3,
 * metal/dflash.metal:13-34) with the destination row taken modulo the
 * buffer: copies rows of the residual stream into one tap of the feature
 * rows, a NaN as 0 and an infinity as +-65504, so no non-finite value
 * reaches the drafter. */
kernel void kernel_dflash_capture_rows(
        constant ds4_metal_args_dflash_capture &args,
        device const float *src,
        device       float *features,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)args.n_rows * args.n_embd;
    if ((uint64_t)gid >= count || args.aux_index >= args.n_aux || args.n_slots == 0u) return;

    const uint row = gid / args.n_embd;
    const uint col = gid - row * args.n_embd;
    float value =
        src[(uint64_t)(args.src_row0 + row) * args.n_embd + col];
    const uint value_bits = as_type<uint>(value);
    if ((value_bits & 0x7f800000u) == 0x7f800000u) {
        value = (value_bits & 0x007fffffu) != 0u ? 0.0f :
            ((value_bits & 0x80000000u) != 0u ? -65504.0f : 65504.0f);
    }
    const uint slot = (args.dst_row0 + row) % args.n_slots;
    const uint64_t dst =
        (uint64_t)slot * args.n_aux * args.n_embd +
        (uint64_t)args.aux_index * args.n_embd + col;
    features[dst] = value;
}

/* --- the drafter's K/V ring ---------------------------------------------- */

struct ds4_metal_args_dflash_ring_write {
    uint32_t n_rows;
    uint32_t pos0;         /* position of row 0; row r goes to slot (pos0 + r) % cache_cap */
    uint32_t cache_cap;
    uint32_t width;        /* n_head_kv * head_dim */
};

/* antirez's kernel_dflash_commit_kv_f16 (laguna-s2.1, 8f620f3,
 * metal/dflash.metal:84-101), which writes K and V rows as f16 at slot
 * position % cache_cap, plus the slot's position tag, which the attention's
 * rule reads (a slot counts only for a query at q from an anchor at p with
 * q - tag < window and tag < p). */
kernel void kernel_dflash_ring_write(
        constant ds4_metal_args_dflash_ring_write &args,
        device const float *k,
        device const float *v,
        device        half *key_cache,
        device        half *value_cache,
        device     int32_t *tags,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)args.n_rows * args.width;
    if ((uint64_t)gid >= count || args.cache_cap == 0u) return;

    const uint row = gid / args.width;
    const uint col = gid - row * args.width;
    const uint cache_row = (args.pos0 + row) % args.cache_cap;
    const uint64_t src = (uint64_t)row * args.width + col;
    const uint64_t dst = (uint64_t)cache_row * args.width + col;
    key_cache[dst] = (half)k[src];
    value_cache[dst] = (half)v[src];
    if (col == 0u) tags[cache_row] = (int32_t)(args.pos0 + row);
}

/* --- q/k head norm with rope ---------------------------------------------- */

struct ds4_metal_args_dflash_head_norm_rope {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;     /* multiple of 32, <= 256 */
    uint32_t rope_dims;    /* even, <= head_dim; pairs (i, i + rope_dims/2) */
    uint32_t pos0;         /* position of row 0; row r sits at pos0 + r */
    float    eps;
    uint32_t pad0;
    uint32_t pad1;
    float    inv_freq[128];/* base^(-2i/rope_dims), rounded once from double on the host */
};

/* y = rope(x * rsqrt(mean(x^2) + eps) * w): llama.cpp's build_norm (RMS) on
 * each head followed by NeoX rope.  The angle is (float)pos * inv_freq[i],
 * as llama.cpp's Metal rope forms it; cos and sin are the precise ones, so
 * the only error past the float angle is its own rounding. */
kernel void kernel_dflash_head_norm_rope(
        constant ds4_metal_args_dflash_head_norm_rope & args,
        device const float *x,         /* [n_rows][n_head][head_dim] */
        device const float *w,         /* [head_dim] */
        device float       *y,         /* [n_rows][n_head][head_dim], may alias x */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x, r = tgpig.y;
    if (h >= args.n_head || r >= args.n_rows) return;
    const uint D = args.head_dim, npt = D / 32u;
    device const float *src = x + ((uint64_t)r * args.n_head + h) * D;
    device float *dst = y + ((uint64_t)r * args.n_head + h) * D;
    threadgroup float row[256];
    float ss = 0.0f;
    for (uint k = 0; k < npt; k++) {
        const float v = src[tiisg + 32u * k];
        row[tiisg + 32u * k] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float scale = 1.0f / sqrt(ss / (float)D + args.eps);
    for (uint k = 0; k < npt; k++) {
        const uint i = tiisg + 32u * k;
        row[i] = row[i] * scale * w[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_dims = args.rope_dims / 2u;
    const float pos = (float)(args.pos0 + r);
    for (uint k = 0; k < npt; k++) {
        const uint i = tiisg + 32u * k;
        float out = row[i];
        if (i < args.rope_dims) {
            const uint pair = i < half_dims ? i : i - half_dims;
            const float theta = pos * args.inv_freq[pair];
            const float c = precise::cos(theta), s = precise::sin(theta);
            out = i < half_dims ? row[i] * c - row[i + half_dims] * s
                                : row[i - half_dims] * s + row[i] * c;
        }
        dst[i] = out;
    }
}

/* --- the DFlash2 block convolution ---------------------------------------- */

struct ds4_metal_args_dflash_conv {
    uint32_t n_rows;
    uint32_t block_rows;   /* rows per drafted block; row 0 of a block has no predecessor */
    uint32_t n_embd;
    uint32_t group;        /* channels per coefficient group */
    uint32_t side;         /* 0: before the sublayer, 1: after it */
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

/* build_dflash2_conv: two taps along the block, tap 0 the row itself and
 * tap 1 the previous row of its block.  The coefficient of channel c for tap
 * k on side s is dyn[row][(s*2 + k)*G + c/group] + base[(s*2 + k)*E + c]
 * (conv_proj's output of the side-0 input row, plus the learned base). */
kernel void kernel_dflash_conv(
        constant ds4_metal_args_dflash_conv & args,
        device const float *x,         /* [n_rows][n_embd] */
        device const float *dyn,       /* [n_rows][2*2*G] conv_proj output */
        device const float *base,      /* [2][2][n_embd] */
        device float       *y,         /* [n_rows][n_embd] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint c = tgpig.x * ntg.x + tpitg.x, t = tgpig.y;
    if (c >= args.n_embd || t >= args.n_rows) return;
    const uint E = args.n_embd, G = E / args.group, g = c / args.group;
    device const float *d = dyn + (uint64_t)t * 4u * G + args.side * 2u * G;
    device const float *b = base + (uint64_t)args.side * 2u * E;
    const float w0 = d[g] + b[c];
    float acc = w0 * x[(uint64_t)t * E + c];
    if ((t % args.block_rows) != 0u) {
        const float w1 = d[G + g] + b[E + c];
        acc += w1 * x[(uint64_t)(t - 1u) * E + c];
    }
    y[(uint64_t)t * E + c] = acc;
}

/* --- windowed attention over the ring and the block ----------------------- */

struct ds4_metal_args_dflash_attn {
    uint32_t n_rows;       /* block rows; row t queries from position pos0 + t */
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;     /* multiple of 32, <= 256 */
    uint32_t ring_slots;
    uint32_t window;       /* a key at kp is dropped for a query at q when q - kp >= window */
    uint32_t pos0;         /* the anchor's position */
    float    scale;
};

#define DFLASH_ATTN_NSG 4

/* One threadgroup per (query head, row), DFLASH_ATTN_NSG simdgroups walking
 * the keys in turn with an online softmax, merged at the end.  Keys are the
 * ring slots whose tag tp is a committed context position (0 <= tp < pos0)
 * inside the row's window, then every row of the block (non-causal, also
 * windowed).  A slot outside the rule is a hole: it costs acceptance, never
 * correctness.  GQA: query head h reads kv head h / (n_head / n_head_kv). */
kernel void kernel_dflash_attn(
        constant ds4_metal_args_dflash_attn & args,
        device const float   *q,        /* [n_rows][n_head][D] */
        device const float   *k_blk,    /* [n_rows][n_head_kv][D] */
        device const float   *v_blk,    /* [n_rows][n_head_kv][D] */
        device const half    *ring_k,   /* [ring_slots][n_head_kv][D] */
        device const half    *ring_v,   /* [ring_slots][n_head_kv][D] */
        device const int32_t *ring_pos, /* [ring_slots] position tags, < 0 empty */
        device float         *out,      /* [n_rows][n_head][D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint h = tgpig.x, t = tgpig.y;
    if (h >= args.n_head || t >= args.n_rows) return;
    const uint D = args.head_dim, npt = D / 32u, Hkv = args.n_head_kv;
    const uint kvh = h / (args.n_head / Hkv);
    const int64_t qpos = (int64_t)args.pos0 + t;
    device const float *qr = q + ((uint64_t)t * args.n_head + h) * D;
    float qv[8], acc[8];
    for (uint k = 0; k < npt; k++) { qv[k] = qr[tiisg + 32u * k] * args.scale; acc[k] = 0.0f; }
    float m = -1.0e30f, l = 0.0f;
    const uint n_keys = args.ring_slots + args.n_rows;
    for (uint key = sgitg; key < n_keys; key += DFLASH_ATTN_NSG) {
        float kv[8], vv[8];
        if (key < args.ring_slots) {
            const int64_t tp = ring_pos[key];
            if (tp < 0 || tp >= (int64_t)args.pos0 || qpos - tp >= (int64_t)args.window) continue;
            device const half *kr = ring_k + ((uint64_t)key * Hkv + kvh) * D;
            device const half *vr = ring_v + ((uint64_t)key * Hkv + kvh) * D;
            for (uint k = 0; k < npt; k++) { kv[k] = (float)kr[tiisg + 32u * k]; vv[k] = (float)vr[tiisg + 32u * k]; }
        } else {
            const uint j = key - args.ring_slots;
            if (qpos - ((int64_t)args.pos0 + j) >= (int64_t)args.window) continue;
            device const float *kr = k_blk + ((uint64_t)j * Hkv + kvh) * D;
            device const float *vr = v_blk + ((uint64_t)j * Hkv + kvh) * D;
            for (uint k = 0; k < npt; k++) { kv[k] = kr[tiisg + 32u * k]; vv[k] = vr[tiisg + 32u * k]; }
        }
        float dot = 0.0f;
        for (uint k = 0; k < npt; k++) dot += qv[k] * kv[k];
        dot = simd_sum(dot);
        const float m_new = max(m, dot);
        const float corr = precise::exp(m - m_new), p = precise::exp(dot - m_new);
        l = l * corr + p;
        for (uint k = 0; k < npt; k++) acc[k] = acc[k] * corr + p * vv[k];
        m = m_new;
    }
    threadgroup float sg_m[DFLASH_ATTN_NSG], sg_l[DFLASH_ATTN_NSG];
    threadgroup float sg_acc[DFLASH_ATTN_NSG][256];
    if (tiisg == 0) { sg_m[sgitg] = m; sg_l[sgitg] = l; }
    for (uint k = 0; k < npt; k++) sg_acc[sgitg][tiisg + 32u * k] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0) return;
    float mt = -1.0e30f;
    for (uint s = 0; s < DFLASH_ATTN_NSG; s++) mt = max(mt, sg_m[s]);
    float lt = 0.0f, o[8] = {0};
    for (uint s = 0; s < DFLASH_ATTN_NSG; s++) {
        if (sg_l[s] == 0.0f) continue;
        const float f = precise::exp(sg_m[s] - mt);
        lt += sg_l[s] * f;
        for (uint k = 0; k < npt; k++) o[k] += sg_acc[s][tiisg + 32u * k] * f;
    }
    device float *dst = out + ((uint64_t)t * args.n_head + h) * D;
    const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
    for (uint k = 0; k < npt; k++) dst[tiisg + 32u * k] = o[k] * inv;
}

/* --- the selector's lattice scores ---------------------------------------- */

struct ds4_metal_args_dflash_selector {
    uint32_t n_rows;       /* block rows; rows 1..n_rows-1 are scored */
    uint32_t top_k;        /* candidates per row, <= 16 */
    uint32_t rank;         /* multiple of 32, <= 1024 */
    uint32_t vocab;
    uint32_t anchor;       /* the block's anchor token, row 1's only predecessor */
    uint32_t pad0;
    uint64_t row_bytes;    /* one Q4_0 selector table row: rank/32 blocks of 18 bytes */
};

/* one element of a Q4_0 row: 18-byte blocks of 32 (f16 scale, 16 nibble
 * bytes, low nibbles first), value d * (q - 8) */
static inline float dflash_q4_0_at(device const uchar *row, uint i) {
    device const uchar *blk = row + (uint64_t)(i / 32u) * 18u;
    const float d = (float)(*(device const half *)blk);
    const uint j = i % 32u;
    const uint q = j < 16u ? (blk[2 + j] & 0xFu) : (blk[2 + j - 16u] >> 4);
    return d * ((float)q - 8.0f);
}

/* build_dflash2_selector: score[t][j][i] = succ[c_t,i] . (pred[p_j] * gate_t)
 * + logit_t[c_t,i], with c_t,i the i-th candidate of row t and p_j the
 * anchor at row 1 (every j alike, as llama.cpp repeats it) or the j-th
 * candidate of row t - 1.  One threadgroup per (j, t); simdgroup i owns
 * candidate i. */
kernel void kernel_dflash_selector(
        constant ds4_metal_args_dflash_selector & args,
        device const int32_t *cand,     /* [n_rows][top_k] */
        device const float   *logits,   /* [n_rows][vocab] */
        device const float   *gate,     /* [n_rows][rank] */
        device const uchar   *pred,     /* [vocab] Q4_0 rows of rank */
        device const uchar   *succ,     /* [vocab] Q4_0 rows of rank */
        device float         *scores,   /* [n_rows][top_k][top_k] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint j = tgpig.x, t = tgpig.y + 1u, K = args.top_k, R = args.rank;
    const uint tid = (uint)sgitg * 32u + tiisg;
    if (j >= K || t >= args.n_rows) return;
    const uint p = t == 1u ? args.anchor : (uint)cand[(uint64_t)(t - 1u) * K + j];
    threadgroup float cond[1024];
    device const uchar *prow = pred + (uint64_t)p * args.row_bytes;
    device const float *g = gate + (uint64_t)t * R;
    for (uint r = tid; r < R; r += 32u * K) cond[r] = dflash_q4_0_at(prow, r) * g[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg >= K) return;
    const uint c = (uint)cand[(uint64_t)t * K + sgitg];
    device const uchar *srow = succ + (uint64_t)c * args.row_bytes;
    float dot = 0.0f;
    for (uint r = tiisg; r < R; r += 32u) dot += dflash_q4_0_at(srow, r) * cond[r];
    dot = simd_sum(dot);
    if (tiisg == 0) {
        scores[((uint64_t)t * K + j) * K + sgitg] = dot + logits[(uint64_t)t * args.vocab + c];
    }
}
