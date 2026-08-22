/*
 * Mellum 2's Metal dispatch, gathered into one file.
 *
 * This is #included by ds4_metal.m rather than compiled separately, and that
 * is deliberate: these functions reach twelve file-static helpers and seven
 * mutable module globals -- g_initialized, the shared flash-attention scratch,
 * and pipeline caches owned jointly with the Laguna and GLM paths.  Compiling
 * this apart would mean exporting that state across a translation-unit
 * boundary, which is a worse arrangement than the one it replaces.  The file
 * split is for reading; the seam it draws is what a later real extraction
 * would have to sever, and it is now visible in one place.
 *
 * Nothing outside these functions calls into them, so they can live at the end
 * of their parent without forward declarations.
 */

/* An override that is off unless explicitly set to something other than 0. */
static int ds4_mellum_env_opt_in(const char *name) {
    const char *env = getenv(name);
    return env && *env && strcmp(env, "0") != 0;
}
/* A default-on policy, disabled only by an explicit 0. */
static int ds4_mellum_env_opt_out(const char *name) {
    const char *env = getenv(name);
    return !(env && *env && strcmp(env, "0") == 0);
}
static uint32_t ds4_mellum_env_u32(const char *name, uint32_t lo, uint32_t hi) {
    const char *env = getenv(name);
    if (!env || !*env) return 0;
    char *end = NULL;
    const unsigned long v = strtoul(env, &end, 10);
    if (end == env || v < lo || v > hi) return 0;
    return (uint32_t)v;
}
const ds4_mellum_runtime *ds4_mellum_runtime_get(void) {
    static ds4_mellum_runtime rt;
    static int resolved = 0;
    if (!resolved) {
        rt.moe_gemm       = ds4_mellum_env_opt_out("DS4_MELLUM_MOE_GEMM");
        rt.prefill_exact  = ds4_mellum_env_opt_out("DS4_MELLUM_PREFILL_EXACT");
        rt.attn_group     = ds4_mellum_env_opt_out("DS4_MELLUM_ATTN_GROUP");
        rt.sync_batch     = ds4_mellum_env_opt_out("DS4_MELLUM_SYNC_BATCH");
        rt.prefill_chunk  = ds4_mellum_env_u32("DS4_MELLUM_PREFILL_CHUNK",
                                               1u, 1u << 20);
        rt.attn_trace     = ds4_mellum_env_opt_in("DS4_MELLUM_ATTN_TRACE");
        rt.sync_trace     = ds4_mellum_env_opt_in("DS4_MELLUM_SYNC_TRACE");
        rt.profile_prefill_tokens = (int)ds4_mellum_env_u32(
            "DS4_MELLUM_PROFILE_PREFILL_TOKENS", 64u, 1u << 20);
        rt.profile_decode_depth = (int)ds4_mellum_env_u32(
            "DS4_MELLUM_PROFILE_DECODE_DEPTH", 1u, 1u << 20);
        resolved = 1;
    }
    return &rt;
}
int ds4_gpu_mellum_attn_group_enabled(void) {
    return ds4_mellum_runtime_get()->attn_group;
}
int ds4_gpu_mellum_gqa_decode_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t                cache_cap,
        uint32_t                key_start,
        uint32_t                key_count,
        uint32_t                n_head,
        uint32_t                n_head_kv,
        uint32_t                head_dim,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !q || !key_cache || !value_cache || cache_cap == 0 ||
        key_count == 0 || key_count > cache_cap || n_head == 0 ||
        n_head_kv == 0 || n_head % n_head_kv != 0 || head_dim != 128u ||
        !isfinite(scale) || scale <= 0.0f) return 0;
    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t kv_bytes = (uint64_t)cache_cap * n_head_kv * head_dim *
            sizeof(uint16_t);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = ds4_gpu_tensor_buffer(value_cache);
        if (!outbuf || !qbuf || !keybuf || !valuebuf ||
            ds4_gpu_tensor_bytes(out) < q_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < kv_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < kv_bytes) {
            fprintf(stderr, "ds4: Metal Mellum GQA received undersized buffers\n");
            return 0;
        }
        /*
         * The serial kernel spends its time in a latency chain, not on the
         * memory bus, so striping the key range over eight SIMD groups is
         * worth close to eight times at depth.  It reassociates the softmax
         * once key_count passes 256, which costs the bitwise decode-equals-
         * prefill contract, so it stays opt-in.
         */
        /*
         * Head grouping supersedes plain split-K when the GQA ratio matches:
         * it keeps the key-range split (over nwg workgroups rather than SIMD
         * groups) and additionally fetches each K/V row once for the whole
         * group.  Partials go through the generic FlashAttention reduce.
         */
        /*
         * Below the threshold the grouped path is all cost: it would dispatch
         * a 32-workgroup kernel plus a 1024-thread reduce to score a handful
         * of keys, and it reassociates the softmax where the serial kernel is
         * still bitwise exact.  Short histories therefore keep the same
         * arithmetic they have always had, which is the property the split
         * kernel advertises and this one should not quietly drop.
         */
        if (ds4_gpu_mellum_attn_group_enabled() &&
            key_count > DS4_MELLUM_ATTN_MIN_KEYS &&
            n_head == n_head_kv * DS4_MELLUM_GROUP_HEADS) {
            const uint32_t ncpsg = 32u;
            const uint32_t nwg = 32u;
            const uint32_t nsg =
                ds4_gpu_flash_attn_vec_nsg(key_count, nwg, ncpsg);
            const NSUInteger nrows = (NSUInteger)n_head;
            const NSUInteger tmp_bytes =
                nrows * head_dim * nwg * sizeof(float) +
                nrows * 2u * nwg * sizeof(float);
            id<MTLComputePipelineState> group_pipeline =
                ds4_gpu_get_pipeline("kernel_mellum_attention_decode_gqa8_split_f16");
            id<MTLComputePipelineState> reduce_pipeline =
                ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim,
                                                       (int32_t)nwg);
            /*
             * This is the default path now, so a missing pipeline or a scratch
             * allocation failure must degrade to the split kernel rather than
             * fail the decode outright.  Anything that fails after work has
             * been submitted still propagates.
             */
            const int grouped_ready =
                group_pipeline != nil && reduce_pipeline != nil &&
                ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                              &g_flash_attn_tmp_bytes,
                                              tmp_bytes,
                                              "ds4_mellum_attn_group_tmp") != 0;
            if (grouped_ready) {
            int group_owned = 0;
            id<MTLCommandBuffer> group_cb =
                ds4_gpu_command_buffer(&group_owned);
            if (!group_cb) return 0;
            const ds4_gpu_mellum_gqa_group_decode_args group_args = {
                .n_head = n_head, .n_head_kv = n_head_kv,
                .head_dim = head_dim, .cache_cap = cache_cap,
                .key_start = key_start, .key_count = key_count,
                .nsg = nsg, .nwg = nwg, .scale = scale,
            };
            const NSUInteger group_shared_bytes =
                (NSUInteger)DS4_MELLUM_GROUP_HEADS * nsg *
                (2u + head_dim) * sizeof(float);
            id<MTLComputeCommandEncoder> genc =
                ds4_gpu_compute_encoder(group_cb);
            [genc setComputePipelineState:group_pipeline];
            [genc setBytes:&group_args length:sizeof(group_args) atIndex:0];
            [genc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [genc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
            [genc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:3];
            [genc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:4];
            [genc setThreadgroupMemoryLength:group_shared_bytes atIndex:0];
            [genc dispatchThreadgroups:MTLSizeMake(n_head_kv, 1, nwg)
                 threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
            ds4_gpu_end_compute_encoder(group_cb, genc);

            ds4_gpu_flash_attn_reduce_args reduce_args = {
                .nrows = (int32_t)nrows,
            };
            id<MTLComputeCommandEncoder> renc =
                ds4_gpu_compute_encoder(group_cb);
            [renc setComputePipelineState:reduce_pipeline];
            [renc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
            [renc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
            [renc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:2];
            [renc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
            ds4_gpu_end_compute_encoder(group_cb, renc);
            if (!ds4_gpu_finish_command_buffer(group_cb, group_owned,
                                               "Mellum GQA grouped decode")) {
                return 0;
            }
            return 1;
            }
            /* Not ready: fall through to the serial kernel below. */
        }

        /*
         * Everything that does not go through head grouping -- short
         * histories, and any GQA ratio the grouped kernel does not serve --
         * runs the serial kernel, which is also the bitwise oracle.
         */
        const char *kernel_name = "kernel_mellum_attention_decode_gqa_f16";
        if (!g_mellum_gqa_decode_pipeline) {
            g_mellum_gqa_decode_pipeline = ds4_gpu_get_pipeline(kernel_name);
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_mellum_gqa_decode_pipeline, kernel_name);
        /* Which kernel ran, reported only on a new maximum key_count: a run
         * that never prints one above the threshold never left the serial
         * path, however many times it was dispatched. */
        if (ds4_mellum_runtime_get()->attn_trace) {
            static uint32_t reported_max = 0;
            if (key_count > reported_max) {
                reported_max = key_count;
                fprintf(stderr,
                        "ds4: Mellum decode attn kernel=%s pipeline=%s max_key_count=%u\n",
                        kernel_name, pipeline ? "ok" : "MISSING", key_count);
            }
        }
        if (!pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_mellum_gqa_decode_args args = {
            .n_head = n_head, .n_head_kv = n_head_kv, .head_dim = head_dim,
            .cache_cap = cache_cap, .key_start = key_start,
            .key_count = key_count, .scale = scale,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum GQA decode")) return 0;
    }
    return 1;
}
int ds4_gpu_mellum_gqa_prefill_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        ds4_gpu_tensor       *staged_key,
        ds4_gpu_tensor       *staged_value,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !key_cache || !value_cache || !staged_key || !staged_value ||
        !q || !k || !v || n_tokens == 0 || n_tokens > cache_cap ||
        pos0 > UINT32_MAX - n_tokens ||
        cache_cap == 0 || n_head == 0 || n_head_kv == 0 ||
        n_head % n_head_kv != 0 || head_dim != 128u ||
        !isfinite(scale) || scale <= 0.0f) return 0;
    @autoreleasepool {
        const uint64_t q_values = (uint64_t)n_tokens * n_head * head_dim;
        const uint64_t kv_values = (uint64_t)n_tokens * n_head_kv * head_dim;
        const uint64_t cache_values =
            (uint64_t)cache_cap * n_head_kv * head_dim;
        if (q_values > NSUIntegerMax / sizeof(float) ||
            kv_values > NSUIntegerMax / sizeof(uint16_t) ||
            ds4_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
            ds4_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
            ds4_gpu_tensor_bytes(v) < kv_values * sizeof(float) ||
            ds4_gpu_tensor_bytes(out) < q_values * sizeof(float) ||
            ds4_gpu_tensor_bytes(staged_key) < kv_values * sizeof(uint16_t) ||
            ds4_gpu_tensor_bytes(staged_value) < kv_values * sizeof(uint16_t) ||
            ds4_gpu_tensor_bytes(key_cache) < cache_values * sizeof(uint16_t) ||
            ds4_gpu_tensor_bytes(value_cache) < cache_values * sizeof(uint16_t)) {
            fprintf(stderr, "ds4: Metal Mellum prefill GQA received undersized buffers\n");
            return 0;
        }
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> stagedkeybuf = ds4_gpu_tensor_buffer(staged_key);
        id<MTLBuffer> stagedvaluebuf = ds4_gpu_tensor_buffer(staged_value);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        if (!g_mellum_gqa_prefill_pipeline) {
            g_mellum_gqa_prefill_pipeline = ds4_gpu_get_pipeline(
                "kernel_mellum_attention_prefill_gqa_f16");
        }
        id<MTLComputePipelineState> attention_pipeline = ds4_gpu_hot_pipeline(
            g_mellum_gqa_prefill_pipeline,
            "kernel_mellum_attention_prefill_gqa_f16");
        if (!outbuf || !keybuf || !valuebuf || !stagedkeybuf ||
            !stagedvaluebuf || !qbuf || !kbuf || !vbuf || !attention_pipeline ||
            !g_laguna_stage_kv_pipeline || !g_laguna_commit_kv_pipeline) return 0;

        const ds4_gpu_laguna_prefill_attention_args args = {
            .n_tokens = n_tokens, .pos0 = pos0, .cache_cap = cache_cap,
            .n_head = n_head, .n_head_kv = n_head_kv, .head_dim = head_dim,
            .scale = scale, .pad0 = 0,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_laguna_stage_kv_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:stagedkeybuf offset:ds4_gpu_tensor_offset(staged_key) atIndex:3];
        [enc setBuffer:stagedvaluebuf offset:ds4_gpu_tensor_offset(staged_value) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attention_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:3];
        [enc setBuffer:stagedkeybuf offset:ds4_gpu_tensor_offset(staged_key) atIndex:4];
        [enc setBuffer:stagedvaluebuf offset:ds4_gpu_tensor_offset(staged_value) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_laguna_commit_kv_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:stagedkeybuf offset:ds4_gpu_tensor_offset(staged_key) atIndex:1];
        [enc setBuffer:stagedvaluebuf offset:ds4_gpu_tensor_offset(staged_value) atIndex:2];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum prefill GQA")) return 0;
    }
    return 1;
}
int ds4_gpu_mellum_store_kv_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t                pos,
        uint32_t                cache_cap,
        uint32_t                n_head_kv,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!key_cache || !value_cache || !k || !v || cache_cap == 0 ||
        n_head_kv == 0 || head_dim != 128u) return 0;
    @autoreleasepool {
        const uint64_t kv_values = (uint64_t)n_head_kv * head_dim;
        const uint64_t cache_bytes = (uint64_t)cache_cap * kv_values * sizeof(uint16_t);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_store_kv_f16_pipeline, "kernel_store_kv_f16");
        if (!keybuf || !valuebuf || !kbuf || !vbuf || !pipeline ||
            ds4_gpu_tensor_bytes(key_cache) < cache_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_bytes ||
            ds4_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
            ds4_gpu_tensor_bytes(v) < kv_values * sizeof(float)) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_store_kv_f16_args args = {
            .cache_cap = cache_cap,
            .cache_row = pos % cache_cap,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum KV store")) return 0;
    }
    return 1;
}
int ds4_gpu_mellum_attention_decode_tensor(
        ds4_gpu_tensor                     *out,
        ds4_gpu_tensor                     *norm,
        ds4_gpu_tensor                     *q,
        ds4_gpu_tensor                     *k,
        ds4_gpu_tensor                     *v,
        ds4_gpu_tensor                     *heads,
        ds4_gpu_tensor                     *projected,
        ds4_gpu_tensor                     *key_cache,
        ds4_gpu_tensor                     *value_cache,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_gpu_mellum_attention_desc *desc,
        const ds4_gpu_tensor               *hidden,
        uint32_t                            pos,
        uint32_t                            cache_cap,
        uint32_t                            key_start,
        uint32_t                            key_count) {
    if (!out || !norm || !q || !k || !v || !heads || !projected ||
        !key_cache || !value_cache || !model_map || !desc || !hidden ||
        desc->n_embd == 0 || desc->n_head == 0 || desc->n_head_kv == 0 ||
        desc->n_head % desc->n_head_kv != 0 || desc->head_dim != 128u ||
        desc->n_rot == 0 || desc->n_rot > desc->head_dim ||
        (desc->n_rot & 1u) != 0 || (desc->n_embd & 31u) != 0 ||
        !isfinite(desc->rms_eps) || desc->rms_eps <= 0.0f ||
        !isfinite(desc->freq_base) || desc->freq_base <= 0.0f ||
        !isfinite(desc->freq_scale) || desc->freq_scale <= 0.0f ||
        !isfinite(desc->rope_ext_factor) ||
        !isfinite(desc->rope_attn_factor) ||
        !isfinite(desc->yarn_beta_fast) ||
        !isfinite(desc->yarn_beta_slow) || cache_cap == 0 ||
        key_count == 0 || key_count > cache_cap ||
        (uint64_t)key_start + key_count - 1u != pos) {
        return 0;
    }

    const uint64_t q_dim = (uint64_t)desc->n_head * desc->head_dim;
    const uint64_t kv_dim = (uint64_t)desc->n_head_kv * desc->head_dim;
    if (q_dim > UINT32_MAX || kv_dim > UINT32_MAX ||
        ds4_gpu_tensor_bytes(hidden) < (uint64_t)desc->n_embd * sizeof(float) ||
        ds4_gpu_tensor_bytes(norm) < (uint64_t)desc->n_embd * sizeof(float) ||
        ds4_gpu_tensor_bytes(q) < q_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(k) < kv_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(v) < kv_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(heads) < q_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(projected) < (uint64_t)desc->n_embd * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)desc->n_embd * sizeof(float)) {
        return 0;
    }

    const float attention_scale = 1.0f / sqrtf((float)desc->head_dim);
    return ds4_gpu_rms_norm_weight_tensor(
               norm, hidden, model_map, model_size, desc->attn_norm_offset,
               desc->n_embd, desc->rms_eps) != 0 &&
           ds4_gpu_matmul_q8_0_tensor(
               q, model_map, model_size, desc->q_offset,
               desc->n_embd, q_dim, norm, 1) != 0 &&
           ds4_gpu_matmul_q8_0_tensor(
               k, model_map, model_size, desc->k_offset,
               desc->n_embd, kv_dim, norm, 1) != 0 &&
           ds4_gpu_matmul_q8_0_tensor(
               v, model_map, model_size, desc->v_offset,
               desc->n_embd, kv_dim, norm, 1) != 0 &&
           ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
               q, k, model_map, model_size,
               desc->q_norm_offset, desc->k_norm_offset,
               1, desc->n_head, desc->n_head_kv, desc->head_dim,
               desc->n_rot, pos, desc->n_ctx_orig,
               desc->freq_base, desc->freq_scale, desc->rope_ext_factor,
               desc->rope_attn_factor, desc->yarn_beta_fast,
               desc->yarn_beta_slow, desc->rms_eps) != 0 &&
           ds4_gpu_mellum_store_kv_tensor(
               key_cache, value_cache, k, v, pos, cache_cap,
               desc->n_head_kv, desc->head_dim) != 0 &&
           ds4_gpu_mellum_gqa_decode_tensor(
               heads, q, key_cache, value_cache, cache_cap, key_start,
               key_count, desc->n_head, desc->n_head_kv, desc->head_dim,
               attention_scale) != 0 &&
           ds4_gpu_matmul_q8_0_tensor(
               projected, model_map, model_size, desc->output_offset,
               q_dim, desc->n_embd, heads, 1) != 0 &&
           ds4_gpu_add_tensor(out, projected, hidden, desc->n_embd) != 0;
}
/*
 * Diagnostic: ds4_gpu_matmul_q8_0_legacy_tensor picks its kernel by token
 * count.  One row uses the F32-accumulating matvec, 17..31 rows use a generic
 * GEMM instantiated on half weights *and* half activations, and >=32 rows use
 * the tensor-op path that dequantizes weight tiles to half.  A Q8_0 weight is
 * d(F16) * q(int8) and needs ~18 mantissa bits, so batched prefill projections
 * carry a relative weight error that decode does not.  This switch forces the
 * row-exact decode kernel for every Mellum prefill projection so that
 * batch-versus-decode drift can be attributed to kernel precision rather than
 * graph semantics.  It validates the single release path and is not a second
 * semantic variant.
 */
/*
 * On by default since layer-major prefill became a session path rather than a
 * diagnostic.  Batched Q8 projections dequantize weights *and* activations to
 * half, and against sequential decode over 1,030 tokens that is worth roughly
 * max_abs 0.89 on logits of scale 21.8; forcing the row-exact decode kernel
 * brings it to 0.019.  It costs about half the prefill throughput, which still
 * leaves layer-major several times faster than the tokenwise path it replaced,
 * so the accuracy is the better trade for a shipped session.  Set
 * DS4_MELLUM_PREFILL_EXACT=0 to measure or ship the faster, looser path.
 */
static bool ds4_gpu_mellum_prefill_exact_projections(void) {
    return ds4_mellum_runtime_get()->prefill_exact != 0;
}
int ds4_gpu_mellum_prefill_exact_projections_enabled(void) {
    return ds4_gpu_mellum_prefill_exact_projections() ? 1 : 0;
}
static int ds4_gpu_mellum_prefill_matmul_q8_0(ds4_gpu_tensor       *out,
                                              const void           *model_map,
                                              uint64_t              model_size,
                                              uint64_t              weight_offset,
                                              uint64_t              in_dim,
                                              uint64_t              out_dim,
                                              const ds4_gpu_tensor *x,
                                              uint32_t              n_tokens) {
    if (ds4_gpu_mellum_prefill_exact_projections()) {
        return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            out, model_map, model_size, weight_offset, in_dim, out_dim, x,
            n_tokens);
    }
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, n_tokens);
}
int ds4_gpu_mellum_attention_prefill_tensor(
        ds4_gpu_tensor                     *out,
        ds4_gpu_tensor                     *norm,
        ds4_gpu_tensor                     *q,
        ds4_gpu_tensor                     *k,
        ds4_gpu_tensor                     *v,
        ds4_gpu_tensor                     *heads,
        ds4_gpu_tensor                     *projected,
        ds4_gpu_tensor                     *key_cache,
        ds4_gpu_tensor                     *value_cache,
        ds4_gpu_tensor                     *staged_key,
        ds4_gpu_tensor                     *staged_value,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_gpu_mellum_attention_desc *desc,
        const ds4_gpu_tensor               *hidden,
        uint32_t                            pos0,
        uint32_t                            n_tokens,
        uint32_t                            cache_cap) {
    if (!out || !norm || !q || !k || !v || !heads || !projected ||
        !key_cache || !value_cache || !staged_key || !staged_value ||
        !model_map || !desc || !hidden || n_tokens == 0 ||
        pos0 > UINT32_MAX - n_tokens || desc->n_embd == 0 ||
        desc->n_head == 0 || desc->n_head_kv == 0 ||
        desc->n_head % desc->n_head_kv != 0 || desc->head_dim != 128u ||
        desc->n_rot == 0 || desc->n_rot > desc->head_dim ||
        (desc->n_rot & 1u) != 0 || (desc->n_embd & 31u) != 0 ||
        !isfinite(desc->rms_eps) || desc->rms_eps <= 0.0f ||
        !isfinite(desc->freq_base) || desc->freq_base <= 0.0f ||
        !isfinite(desc->freq_scale) || desc->freq_scale <= 0.0f ||
        !isfinite(desc->rope_ext_factor) ||
        !isfinite(desc->rope_attn_factor) ||
        !isfinite(desc->yarn_beta_fast) || !isfinite(desc->yarn_beta_slow) ||
        cache_cap == 0 || n_tokens > cache_cap) return 0;
    const uint64_t q_dim = (uint64_t)desc->n_head * desc->head_dim;
    const uint64_t kv_dim = (uint64_t)desc->n_head_kv * desc->head_dim;
    const uint64_t hidden_values = (uint64_t)n_tokens * desc->n_embd;
    const uint64_t q_values = (uint64_t)n_tokens * q_dim;
    const uint64_t kv_values = (uint64_t)n_tokens * kv_dim;
    if (q_dim > UINT32_MAX || kv_dim > UINT32_MAX ||
        hidden_values > UINT32_MAX ||
        ds4_gpu_tensor_bytes(hidden) < hidden_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(norm) < hidden_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(v) < kv_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(heads) < q_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(projected) < hidden_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < hidden_values * sizeof(float)) return 0;
    const float attention_scale = 1.0f / sqrtf((float)desc->head_dim);
    return ds4_gpu_rms_norm_weight_rows_tensor(
               norm, hidden, model_map, model_size, desc->attn_norm_offset,
               desc->n_embd, n_tokens, desc->rms_eps) != 0 &&
           ds4_gpu_mellum_prefill_matmul_q8_0(q, model_map, model_size,
                                              desc->q_offset, desc->n_embd,
                                              q_dim, norm, n_tokens) != 0 &&
           ds4_gpu_mellum_prefill_matmul_q8_0(k, model_map, model_size,
                                              desc->k_offset, desc->n_embd,
                                              kv_dim, norm, n_tokens) != 0 &&
           ds4_gpu_mellum_prefill_matmul_q8_0(v, model_map, model_size,
                                              desc->v_offset, desc->n_embd,
                                              kv_dim, norm, n_tokens) != 0 &&
           ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
               q, k, model_map, model_size, desc->q_norm_offset,
               desc->k_norm_offset, n_tokens, desc->n_head, desc->n_head_kv,
               desc->head_dim, desc->n_rot, pos0, desc->n_ctx_orig,
               desc->freq_base, desc->freq_scale, desc->rope_ext_factor,
               desc->rope_attn_factor, desc->yarn_beta_fast,
               desc->yarn_beta_slow, desc->rms_eps) != 0 &&
           ds4_gpu_mellum_gqa_prefill_tensor(
               heads, key_cache, value_cache, staged_key, staged_value, q, k,
               v, pos0, n_tokens, cache_cap, desc->n_head, desc->n_head_kv,
               desc->head_dim, attention_scale) != 0 &&
           ds4_gpu_mellum_prefill_matmul_q8_0(projected, model_map, model_size,
                                              desc->output_offset, q_dim,
                                              desc->n_embd, heads,
                                              n_tokens) != 0 &&
           ds4_gpu_add_tensor(out, projected, hidden,
                              (uint32_t)hidden_values) != 0;
}
int ds4_gpu_mellum_q8_0_layer_decode_tensor(
        ds4_gpu_tensor                         *out,
        ds4_gpu_tensor                         *attention_out,
        ds4_gpu_tensor                         *attention_norm,
        ds4_gpu_tensor                         *q,
        ds4_gpu_tensor                         *k,
        ds4_gpu_tensor                         *v,
        ds4_gpu_tensor                         *heads,
        ds4_gpu_tensor                         *projected,
        ds4_gpu_tensor                         *key_cache,
        ds4_gpu_tensor                         *value_cache,
        ds4_gpu_tensor                         *ffn_norm,
        ds4_gpu_tensor                         *router_logits,
        ds4_gpu_tensor                         *router_selected,
        ds4_gpu_tensor                         *router_weights,
        ds4_gpu_tensor                         *router_probs,
        ds4_gpu_tensor                         *moe_mid,
        ds4_gpu_tensor                         *moe_out,
        const void                             *model_map,
        uint64_t                                model_size,
        const ds4_gpu_mellum_q8_0_layer_desc  *desc,
        const ds4_gpu_tensor                   *hidden,
        uint32_t                                pos,
        uint32_t                                cache_cap,
        uint32_t                                key_start,
        uint32_t                                key_count) {
    if (!out || !attention_out || !attention_norm || !q || !k || !v ||
        !heads || !projected || !key_cache || !value_cache || !ffn_norm ||
        !router_logits || !router_selected || !router_weights || !router_probs ||
        !moe_mid || !moe_out || !model_map || !desc || !hidden ||
        desc->attention.n_embd == 0 || desc->expert_mid_dim == 0 ||
        desc->n_expert == 0 || desc->n_expert > 256u ||
        desc->n_expert_used == 0 || desc->n_expert_used > desc->n_expert ||
        ds4_gpu_tensor_bytes(out) <
            (uint64_t)desc->attention.n_embd * sizeof(float) ||
        ds4_gpu_tensor_bytes(attention_out) <
            (uint64_t)desc->attention.n_embd * sizeof(float) ||
        ds4_gpu_tensor_bytes(ffn_norm) <
            (uint64_t)desc->attention.n_embd * sizeof(float) ||
        ds4_gpu_tensor_bytes(router_logits) <
            (uint64_t)desc->n_expert * sizeof(float) ||
        ds4_gpu_tensor_bytes(router_selected) <
            (uint64_t)desc->n_expert_used * sizeof(int32_t) ||
        ds4_gpu_tensor_bytes(router_weights) <
            (uint64_t)desc->n_expert_used * sizeof(float) ||
        ds4_gpu_tensor_bytes(router_probs) <
            (uint64_t)desc->n_expert * sizeof(float) ||
        ds4_gpu_tensor_bytes(moe_mid) <
            (uint64_t)desc->n_expert_used * desc->expert_mid_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(moe_out) <
            (uint64_t)desc->attention.n_embd * sizeof(float)) {
        return 0;
    }

    const uint32_t n_embd = desc->attention.n_embd;
    return ds4_gpu_mellum_attention_decode_tensor(
               attention_out, attention_norm, q, k, v, heads, projected,
               key_cache, value_cache, model_map, model_size, &desc->attention,
               hidden, pos, cache_cap, key_start, key_count) != 0 &&
           ds4_gpu_rms_norm_weight_tensor(
               ffn_norm, attention_out, model_map, model_size,
               desc->ffn_norm_offset, n_embd, desc->attention.rms_eps) != 0 &&
           (desc->router_is_f32 ?
                ds4_gpu_matmul_f32_tensor(
                    router_logits, model_map, model_size, desc->router_offset,
                    n_embd, desc->n_expert, ffn_norm, 1) :
                ds4_gpu_matmul_q8_0_tensor(
                    router_logits, model_map, model_size, desc->router_offset,
                    n_embd, desc->n_expert, ffn_norm, 1)) != 0 &&
           ds4_gpu_mellum_router_select_tensor(
               router_selected, router_weights, router_probs, router_logits,
               desc->n_expert, desc->n_expert_used) != 0 &&
           ds4_gpu_mellum_q8_0_routed_moe_one_tensor(
               moe_out, moe_mid, model_map, model_size, desc->gate_offset,
               desc->up_offset, desc->down_offset, desc->gate_expert_bytes,
               desc->gate_row_bytes, desc->down_expert_bytes,
               desc->down_row_bytes, n_embd, desc->expert_mid_dim, n_embd,
               router_selected, router_weights, desc->n_expert,
               desc->n_expert_used, ffn_norm) != 0 &&
           ds4_gpu_add_tensor(out, attention_out, moe_out, n_embd) != 0;
}
int ds4_gpu_mellum_q8_0_layer_prefill_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *attention_out,
        ds4_gpu_tensor *attention_norm, ds4_gpu_tensor *q, ds4_gpu_tensor *k,
        ds4_gpu_tensor *v, ds4_gpu_tensor *heads, ds4_gpu_tensor *projected,
        ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache,
        ds4_gpu_tensor *staged_key, ds4_gpu_tensor *staged_value,
        ds4_gpu_tensor *ffn_norm, ds4_gpu_tensor *router_logits,
        ds4_gpu_tensor *router_selected, ds4_gpu_tensor *router_weights,
        ds4_gpu_tensor *router_probs, ds4_gpu_tensor *moe_mid,
        ds4_gpu_tensor *moe_out, const void *model_map, uint64_t model_size,
        const ds4_gpu_mellum_q8_0_layer_desc *desc,
        const ds4_gpu_tensor *hidden, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap) {
    if (!out || !attention_out || !attention_norm || !q || !k || !v || !heads ||
        !projected || !key_cache || !value_cache || !staged_key || !staged_value ||
        !ffn_norm || !router_logits || !router_selected || !router_weights ||
        !router_probs || !moe_mid || !moe_out || !model_map || !desc || !hidden ||
        n_tokens == 0 || desc->attention.n_embd == 0 ||
        desc->expert_mid_dim == 0 || desc->n_expert == 0 ||
        desc->n_expert_used == 0 || desc->n_expert_used > desc->n_expert ||
        n_tokens > UINT32_MAX / desc->attention.n_embd) return 0;
    const uint32_t n_embd = desc->attention.n_embd;
    const uint64_t hidden_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)n_tokens * desc->n_expert * sizeof(float);
    const uint64_t route_bytes = (uint64_t)n_tokens * desc->n_expert_used;
    if (route_bytes > UINT64_MAX / desc->expert_mid_dim ||
        route_bytes > UINT64_MAX / sizeof(int32_t) ||
        route_bytes > UINT64_MAX / sizeof(float) ||
        route_bytes * desc->expert_mid_dim > UINT64_MAX / sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < hidden_bytes ||
        ds4_gpu_tensor_bytes(attention_out) < hidden_bytes ||
        ds4_gpu_tensor_bytes(ffn_norm) < hidden_bytes ||
        ds4_gpu_tensor_bytes(moe_out) < hidden_bytes ||
        ds4_gpu_tensor_bytes(router_logits) < logits_bytes ||
        ds4_gpu_tensor_bytes(router_selected) < route_bytes * sizeof(int32_t) ||
        ds4_gpu_tensor_bytes(router_weights) < route_bytes * sizeof(float) ||
        ds4_gpu_tensor_bytes(router_probs) < logits_bytes ||
        ds4_gpu_tensor_bytes(moe_mid) <
            route_bytes * desc->expert_mid_dim * sizeof(float)) return 0;
    return ds4_gpu_mellum_attention_prefill_tensor(
               attention_out, attention_norm, q, k, v, heads, projected,
               key_cache, value_cache, staged_key, staged_value, model_map,
               model_size, &desc->attention, hidden, pos0, n_tokens,
               cache_cap) != 0 &&
           ds4_gpu_rms_norm_weight_rows_tensor(
               ffn_norm, attention_out, model_map, model_size, desc->ffn_norm_offset,
               n_embd, n_tokens, desc->attention.rms_eps) != 0 &&
           (desc->router_is_f32 ?
                ds4_gpu_matmul_f32_tensor(router_logits, model_map, model_size,
                                          desc->router_offset, n_embd,
                                          desc->n_expert, ffn_norm, n_tokens) :
                ds4_gpu_matmul_q8_0_tensor(router_logits, model_map, model_size,
                                           desc->router_offset, n_embd,
                                           desc->n_expert, ffn_norm, n_tokens)) != 0 &&
           ds4_gpu_mellum_router_select_batch_tensor(
               router_selected, router_weights, router_probs, router_logits,
               desc->n_expert, desc->n_expert_used, n_tokens) != 0 &&
           ds4_gpu_mellum_q8_0_routed_moe_batch_tensor(
               moe_out, moe_mid, model_map, model_size, desc->gate_offset,
               desc->up_offset, desc->down_offset, desc->gate_expert_bytes,
               desc->gate_row_bytes, desc->down_expert_bytes,
               desc->down_row_bytes, n_embd, desc->expert_mid_dim, n_embd,
               router_selected, router_weights, desc->n_expert,
               desc->n_expert_used, ffn_norm, n_tokens) != 0 &&
           ds4_gpu_add_tensor(out, attention_out, moe_out,
                              n_tokens * n_embd) != 0;
}
int ds4_gpu_mellum_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || n_expert == 0 ||
        n_expert > 256u || n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }
    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal Mellum router received undersized buffers\n");
            return 0;
        }
        if (!g_mellum_router_select_one_pipeline) {
            g_mellum_router_select_one_pipeline =
                ds4_gpu_get_pipeline("kernel_mellum_router_select_one");
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_mellum_router_select_one_pipeline, "kernel_mellum_router_select_one");
        if (!pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_mellum_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:3];
        [enc setBuffer:probsbuf offset:ds4_gpu_tensor_offset(probs) atIndex:4];
        [enc setThreadgroupMemoryLength:512u * sizeof(float) +
                                          256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum router select")) return 0;
    }
    return 1;
}
int ds4_gpu_mellum_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || n_expert == 0 ||
        n_expert > 256u || n_expert_used == 0 || n_expert_used > n_expert ||
        n_tokens == 0) return 0;
    const uint64_t logits_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)n_tokens * n_expert_used * sizeof(float);
    const uint64_t probs_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < logits_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes ||
            ds4_gpu_tensor_bytes(probs) < probs_bytes) {
            fprintf(stderr, "ds4: Metal Mellum batch router received undersized buffers\n");
            return 0;
        }
        if (!g_mellum_router_select_batch_pipeline) {
            g_mellum_router_select_batch_pipeline =
                ds4_gpu_get_pipeline("kernel_mellum_router_select_batch");
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_mellum_router_select_batch_pipeline,
            "kernel_mellum_router_select_batch");
        if (!pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_mellum_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:3];
        [enc setBuffer:probsbuf offset:ds4_gpu_tensor_offset(probs) atIndex:4];
        [enc setThreadgroupMemoryLength:512u * sizeof(float) +
                                          256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum batch router select"))
            return 0;
    }
    return 1;
}
int ds4_gpu_mellum_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_total_expert == 0 || n_expert == 0 || n_expert > n_total_expert ||
        /* Down scratch is n_expert * 256 floats; keep it inside 32 KiB. */
        n_expert > 32u ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        (expert_in_dim % 256u) != 0 || (expert_mid_dim % 32u) != 0 ||
        gate_row_bytes != (uint64_t)(expert_in_dim / 256u) * 144u ||
        down_row_bytes != (uint64_t)(expert_mid_dim / 32u) * 34u ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: Metal Mellum routed MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || gate_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_bytes > model_size - up_offset ||
        down_offset > model_size || down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: Metal Mellum routed MoE tensor range is outside the mapped model\n");
        return 0;
    }

    @autoreleasepool {
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes =
            (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal Mellum routed MoE received undersized activation buffers\n");
            return 0;
        }

        uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
        id<MTLBuffer> gatebuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, gate_offset, gate_bytes, &gate_inner);
        id<MTLBuffer> upbuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, up_offset, gate_bytes, &up_inner);
        id<MTLBuffer> downbuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, down_offset, down_bytes, &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;

        id<MTLComputePipelineState> pair_pipeline = ds4_gpu_hot_pipeline(
            g_glm_q4_k_pair_swiglu_f32_pipeline,
            "kernel_glm_q4_K_pair_swiglu_f32");
        if (!g_mellum_q8_0_down_f32_pipeline) {
            g_mellum_q8_0_down_f32_pipeline =
                ds4_gpu_get_pipeline("kernel_mellum_q8_0_down_f32");
        }
        id<MTLComputePipelineState> down_pipeline = ds4_gpu_hot_pipeline(
            g_mellum_q8_0_down_f32_pipeline, "kernel_mellum_q8_0_down_f32");
        if (!pair_pipeline || !down_pipeline) return 0;

        ds4_gpu_glm_routed_moe_args args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = DS4_METAL_TENSOR_Q8_0,
            .tp_rank = 0,
            .tp_world = 1,
            .tp_expert_base = 0,
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = gate_expert_bytes,
            .up_row_bytes = gate_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:upbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc useResource:gatebuf usage:MTLResourceUsageRead];
        [enc useResource:upbuf usage:MTLResourceUsageRead];
        [enc setThreadgroupMemoryLength:512u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)expert_mid_dim,
                                              (NSUInteger)n_expert, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:downbuf offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc useResource:downbuf usage:MTLResourceUsageRead];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_expert * 256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)out_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum routed MoE")) return 0;
    }
    return 1;
}
int ds4_gpu_mellum_q8_0_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_total_expert == 0 || n_expert == 0 || n_expert > n_total_expert ||
        /* Down scratch is n_expert * 256 floats; keep it inside 32 KiB. */
        n_expert > 32u ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        (expert_in_dim % 32u) != 0 || (expert_mid_dim % 32u) != 0 ||
        gate_row_bytes != (uint64_t)(expert_in_dim / 32u) * 34u ||
        down_row_bytes != (uint64_t)(expert_mid_dim / 32u) * 34u ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: Metal Mellum Q8_0 routed MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || gate_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_bytes > model_size - up_offset ||
        down_offset > model_size || down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: Metal Mellum Q8_0 routed MoE tensor range is outside the mapped model\n");
        return 0;
    }

    @autoreleasepool {
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes =
            (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal Mellum Q8_0 routed MoE received undersized activation buffers\n");
            return 0;
        }

        uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
        id<MTLBuffer> gatebuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, gate_offset, gate_bytes, &gate_inner);
        id<MTLBuffer> upbuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, up_offset, gate_bytes, &up_inner);
        id<MTLBuffer> downbuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, down_offset, down_bytes, &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;

        if (!g_mellum_q8_0_pair_swiglu_f32_pipeline) {
            g_mellum_q8_0_pair_swiglu_f32_pipeline =
                ds4_gpu_get_pipeline("kernel_mellum_q8_0_pair_swiglu_f32");
        }
        if (!g_mellum_q8_0_down_f32_pipeline) {
            g_mellum_q8_0_down_f32_pipeline =
                ds4_gpu_get_pipeline("kernel_mellum_q8_0_down_f32");
        }
        id<MTLComputePipelineState> pair_pipeline = ds4_gpu_hot_pipeline(
            g_mellum_q8_0_pair_swiglu_f32_pipeline,
            "kernel_mellum_q8_0_pair_swiglu_f32");
        id<MTLComputePipelineState> down_pipeline = ds4_gpu_hot_pipeline(
            g_mellum_q8_0_down_f32_pipeline, "kernel_mellum_q8_0_down_f32");
        if (!pair_pipeline || !down_pipeline) return 0;

        ds4_gpu_glm_routed_moe_args args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = DS4_METAL_TENSOR_Q8_0,
            .tp_rank = 0,
            .tp_world = 1,
            .tp_expert_base = 0,
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = gate_expert_bytes,
            .up_row_bytes = gate_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:upbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc useResource:gatebuf usage:MTLResourceUsageRead];
        [enc useResource:upbuf usage:MTLResourceUsageRead];
        [enc setThreadgroupMemoryLength:512u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)expert_mid_dim,
                                              (NSUInteger)n_expert, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:downbuf offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc useResource:downbuf usage:MTLResourceUsageRead];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_expert * 256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)out_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum Q8_0 routed MoE")) {
            return 0;
        }
    }
    return 1;
}
/*
 * Expert-major simdgroup MoE.  Cuts per-layer expert weight traffic from
 * n_tokens * 8 rows to one row per expert per chunk, at the cost of bitwise
 * identity with decode: simd_sum reassociates the reduction.  Off by default;
 * the batch and decode paths remain the bitwise oracle.
 */
/*
 * Expert-major MoE is the production path.  It reassociates, so it is not
 * bitwise against sequential decode -- but the deviation is 0.09% relative rms
 * on logits, inside the model's own 0.26% error against FP32, and five greedy
 * transcripts totalling 531 tokens over 109-token prompts came back
 * byte-identical with it on and off.  DS4_MELLUM_MOE_GEMM=0 selects the
 * bitwise path, which exists as a correctness oracle for the tests rather than
 * as a second configuration to ship.
 */
int ds4_gpu_mellum_moe_gemm_enabled(void) {
    return ds4_mellum_runtime_get()->moe_gemm;
}
/*
 * Bucket the chunk's (token, slot) pairs by selected expert, then hand the
 * caller the buffers its grouped kernel needs.  Expert weights are addressed
 * through an offset table so the same kernel can later be pointed at streamed
 * cache slots instead of model offsets.
 */
static bool ds4_gpu_mellum_moe_group_begin(ds4_gpu_mellum_moe_group *g,
                                           id<MTLCommandBuffer> cb,
                                           uint32_t n_total_expert,
                                           uint32_t n_expert_used,
                                           uint32_t n_tokens,
                                           uint64_t gate_expert_bytes,
                                           uint64_t down_expert_bytes,
                                           uint32_t out_dim,
                                           id<MTLBuffer> selectedbuf,
                                           uint64_t selected_offset) {
    static ds4_gpu_tensor *s_counts, *s_pairs, *s_gate_off, *s_up_off;
    static ds4_gpu_tensor *s_down_off, *s_partial;
    static uint32_t s_experts, s_cap, s_pairs_experts;
    static uint64_t s_expert_bytes, s_down_expert_bytes, s_partial_bytes;

    if (!g || !cb || !selectedbuf || n_total_expert == 0 || n_tokens == 0 ||
        n_expert_used == 0 || n_expert_used > n_total_expert) return false;
    /* One token contributes at most one pair per expert. */
    const uint32_t cap = n_tokens;
    if (cap > UINT32_MAX / n_total_expert) return false;

    if (!s_counts || s_experts != n_total_expert) {
        if (s_counts && ds4_gpu_commands_active()) return false;
        ds4_gpu_tensor_free(s_counts);
        ds4_gpu_tensor_free(s_gate_off);
        ds4_gpu_tensor_free(s_up_off);
        s_counts = ds4_gpu_tensor_alloc((uint64_t)n_total_expert * sizeof(uint32_t));
        ds4_gpu_tensor_free(s_down_off);
        s_gate_off = ds4_gpu_tensor_alloc((uint64_t)n_total_expert * sizeof(uint64_t));
        s_up_off = ds4_gpu_tensor_alloc((uint64_t)n_total_expert * sizeof(uint64_t));
        s_down_off = ds4_gpu_tensor_alloc((uint64_t)n_total_expert * sizeof(uint64_t));
        s_experts = n_total_expert;
        s_expert_bytes = 0;
        s_down_expert_bytes = 0;
    }
    if (!s_pairs || s_cap < cap || s_pairs_experts != n_total_expert) {
        /*
         * Freeing here while a caller-owned batch is open could release a
         * buffer the queued command buffer still references.  Callers keep the
         * shape fixed for a whole chunk, so this only guards a future one that
         * does not.
         */
        if (s_pairs && ds4_gpu_commands_active()) return false;
        ds4_gpu_tensor_free(s_pairs);
        s_pairs = ds4_gpu_tensor_alloc((uint64_t)n_total_expert * cap *
                                       sizeof(uint32_t));
        s_cap = cap;
        s_pairs_experts = n_total_expert;
    }
    if (!s_counts || !s_pairs || !s_gate_off || !s_up_off || !s_down_off) return false;

    if (s_down_expert_bytes != down_expert_bytes) {
        uint64_t *off = malloc((size_t)n_total_expert * sizeof(*off));
        if (!off) return false;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            off[e] = (uint64_t)e * down_expert_bytes;
        }
        const uint64_t bytes = (uint64_t)n_total_expert * sizeof(*off);
        const bool ok = ds4_gpu_tensor_write(s_down_off, 0, off, bytes) != 0;
        free(off);
        if (!ok) return false;
        s_down_expert_bytes = down_expert_bytes;
    }

    /*
     * Per-(token, slot) staging for the expert-major down projection.  Only
     * allocated when that path is on: it is n_tokens * n_expert_used * out_dim
     * floats, about 75 MiB at a 1,024-token chunk.
     */
    if (ds4_gpu_mellum_moe_gemm_enabled() && out_dim > 0) {
        const uint64_t want = (uint64_t)n_tokens * n_expert_used * out_dim *
                              sizeof(float);
        if (!s_partial || s_partial_bytes < want) {
            if (s_partial && ds4_gpu_commands_active()) return false;
            ds4_gpu_tensor_free(s_partial);
            s_partial = ds4_gpu_tensor_alloc(want);
            s_partial_bytes = s_partial ? want : 0;
        }
        if (!s_partial) return false;
    }

    if (s_expert_bytes != gate_expert_bytes) {
        uint64_t *off = malloc((size_t)n_total_expert * sizeof(*off));
        if (!off) return false;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            off[e] = (uint64_t)e * gate_expert_bytes;
        }
        const uint64_t bytes = (uint64_t)n_total_expert * sizeof(*off);
        const bool ok = ds4_gpu_tensor_write(s_gate_off, 0, off, bytes) != 0 &&
                        ds4_gpu_tensor_write(s_up_off, 0, off, bytes) != 0;
        free(off);
        if (!ok) return false;
        s_expert_bytes = gate_expert_bytes;
    }

    if (!g_mellum_moe_bucket_reset_pipeline) {
        g_mellum_moe_bucket_reset_pipeline =
            ds4_gpu_get_pipeline("kernel_mellum_moe_bucket_reset");
    }
    if (!g_mellum_moe_bucket_build_pipeline) {
        g_mellum_moe_bucket_build_pipeline =
            ds4_gpu_get_pipeline("kernel_mellum_moe_bucket_build");
    }
    if (!g_mellum_moe_bucket_reset_pipeline ||
        !g_mellum_moe_bucket_build_pipeline) return false;

    g->args = (ds4_gpu_mellum_moe_group_args){
        .n_tokens = n_tokens,
        .n_expert_used = n_expert_used,
        .n_total_expert = n_total_expert,
        .bucket_cap = s_cap,
    };
    g->counts = s_counts;
    g->pairs = s_pairs;
    g->gate_offsets = s_gate_off;
    g->up_offsets = s_up_off;
    g->down_offsets = s_down_off;
    g->partial = s_partial;

    /* Reset and build are separate encoders: the build must observe the reset. */
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mellum_moe_bucket_reset_pipeline];
    [enc setBytes:&g->args length:sizeof(g->args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(s_counts)
            offset:ds4_gpu_tensor_offset(s_counts) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((n_total_expert + 63u) / 64u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    const uint32_t pair_total = n_tokens * n_expert_used;
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mellum_moe_bucket_build_pipeline];
    [enc setBytes:&g->args length:sizeof(g->args) atIndex:0];
    [enc setBuffer:selectedbuf offset:(NSUInteger)selected_offset atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(s_counts)
            offset:ds4_gpu_tensor_offset(s_counts) atIndex:2];
    [enc setBuffer:ds4_gpu_tensor_buffer(s_pairs)
            offset:ds4_gpu_tensor_offset(s_pairs) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((pair_total + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return true;
}
int ds4_gpu_mellum_q8_0_routed_moe_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *mid, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t expert_in_dim,
        uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_total_expert == 0 || n_expert == 0 || n_expert > n_total_expert ||
        /* Down scratch is n_expert * 256 floats; keep it inside 32 KiB. */
        n_expert > 32u ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 || n_tokens == 0 ||
        n_expert > UINT32_MAX / expert_mid_dim ||
        (expert_in_dim % 32u) != 0 || (expert_mid_dim % 32u) != 0 ||
        gate_row_bytes != (uint64_t)(expert_in_dim / 32u) * 34u ||
        down_row_bytes != (uint64_t)(expert_mid_dim / 32u) * 34u ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) return 0;
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || gate_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_bytes > model_size - up_offset ||
        down_offset > model_size || down_bytes > model_size - down_offset) return 0;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t route_values = (uint64_t)n_tokens * n_expert;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (x_values > UINT64_MAX / sizeof(float) ||
        route_values > UINT64_MAX / expert_mid_dim ||
        route_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float)) return 0;
    const uint64_t mid_values = route_values * expert_mid_dim;
    if (mid_values > UINT64_MAX / sizeof(float) ||
        route_values > UINT64_MAX / sizeof(int32_t) ||
        !ds4_gpu_tensor_buffer(x) || !ds4_gpu_tensor_buffer(mid) ||
        !ds4_gpu_tensor_buffer(out) || !ds4_gpu_tensor_buffer(selected) ||
        !ds4_gpu_tensor_buffer(weights) ||
        ds4_gpu_tensor_bytes(x) < x_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(mid) < mid_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < out_values * sizeof(float) ||
        ds4_gpu_tensor_bytes(selected) < route_values * sizeof(int32_t) ||
        ds4_gpu_tensor_bytes(weights) < route_values * sizeof(float)) return 0;
    @autoreleasepool {
        uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
        id<MTLBuffer> gatebuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, gate_offset, gate_bytes, &gate_inner);
        id<MTLBuffer> upbuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, up_offset, gate_bytes, &up_inner);
        id<MTLBuffer> downbuf = ds4_gpu_wrap_model_exact_range(
            model_map, model_size, down_offset, down_bytes, &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;
        if (!g_mellum_q8_0_pair_swiglu_batch_f32_pipeline)
            g_mellum_q8_0_pair_swiglu_batch_f32_pipeline = ds4_gpu_get_pipeline(
                "kernel_mellum_q8_0_pair_swiglu_batch_f32");
        if (!g_mellum_q8_0_down_batch_f32_pipeline)
            g_mellum_q8_0_down_batch_f32_pipeline = ds4_gpu_get_pipeline(
                "kernel_mellum_q8_0_down_batch_f32");
        id<MTLComputePipelineState> pair_pipeline = ds4_gpu_hot_pipeline(
            g_mellum_q8_0_pair_swiglu_batch_f32_pipeline,
            "kernel_mellum_q8_0_pair_swiglu_batch_f32");
        id<MTLComputePipelineState> down_pipeline = ds4_gpu_hot_pipeline(
            g_mellum_q8_0_down_batch_f32_pipeline,
            "kernel_mellum_q8_0_down_batch_f32");
        if (!pair_pipeline || !down_pipeline) return 0;
        id<MTLComputePipelineState> gemm_pair_pipeline = nil;
        id<MTLComputePipelineState> gemm_down_pipeline = nil;
        id<MTLComputePipelineState> gemm_reduce_pipeline = nil;
        unsigned gemm_rows = 0;
        if (ds4_gpu_mellum_moe_gemm_enabled()) {
            if (!g_mellum_pair_swiglu_gemm_pipeline)
                g_mellum_pair_swiglu_gemm_pipeline = ds4_gpu_get_pipeline(
                    "kernel_mellum_q8_0_pair_swiglu_gemm_f32");
            if (!g_mellum_slot_reduce_pipeline)
                g_mellum_slot_reduce_pipeline = ds4_gpu_get_pipeline(
                    "kernel_mellum_moe_slot_reduce_f32");
            /* R=8 stages 8 * mid_dim floats; fall back to 4 if unavailable. */
            if (!g_mellum_down_grouped8_pipeline)
                g_mellum_down_grouped8_pipeline = ds4_gpu_get_pipeline(
                    "kernel_mellum_q8_0_down_grouped8_f32");
            if (!g_mellum_down_grouped4_pipeline)
                g_mellum_down_grouped4_pipeline = ds4_gpu_get_pipeline(
                    "kernel_mellum_q8_0_down_grouped4_f32");
            gemm_pair_pipeline = ds4_gpu_hot_pipeline(
                g_mellum_pair_swiglu_gemm_pipeline,
                "kernel_mellum_q8_0_pair_swiglu_gemm_f32");
            gemm_reduce_pipeline = ds4_gpu_hot_pipeline(
                g_mellum_slot_reduce_pipeline,
                "kernel_mellum_moe_slot_reduce_f32");
            const uint64_t stage8 = 8ull * expert_mid_dim * sizeof(float);
            if (stage8 <= 32768ull && g_mellum_down_grouped8_pipeline) {
                gemm_down_pipeline = ds4_gpu_hot_pipeline(
                    g_mellum_down_grouped8_pipeline,
                    "kernel_mellum_q8_0_down_grouped8_f32");
                gemm_rows = 8;
            }
            if (!gemm_down_pipeline) {
                gemm_down_pipeline = ds4_gpu_hot_pipeline(
                    g_mellum_down_grouped4_pipeline,
                    "kernel_mellum_q8_0_down_grouped4_f32");
                gemm_rows = gemm_down_pipeline ? 4 : 0;
            }
        }
        const bool want_gemm = gemm_pair_pipeline && gemm_down_pipeline &&
                               gemm_reduce_pipeline && gemm_rows;
        ds4_gpu_glm_routed_moe_args args = {
            .in_dim = expert_in_dim, .mid_dim = expert_mid_dim, .out_dim = out_dim,
            .n_total_expert = n_total_expert, .n_expert_used = n_expert,
            .n_tokens = n_tokens, .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = DS4_METAL_TENSOR_Q8_0, .tp_world = 1,
            .gate_expert_bytes = gate_expert_bytes, .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = gate_expert_bytes, .up_row_bytes = gate_row_bytes,
            .down_expert_bytes = down_expert_bytes, .down_row_bytes = down_row_bytes,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        ds4_gpu_mellum_moe_group ggroup = {0};
        /*
         * Bucketing tokens by expert is what the GEMM path consumes; it is no
         * longer shared with anything else, so it is built only for that.
         */
        const bool grouped = want_gemm &&
            ds4_gpu_mellum_moe_group_begin(&ggroup, cb, n_total_expert, n_expert,
                                            n_tokens, gate_expert_bytes,
                                            down_expert_bytes, out_dim,
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected));
        const bool gemm = grouped && ggroup.partial;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (gemm) {
            [enc setComputePipelineState:gemm_pair_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBytes:&ggroup.args length:sizeof(ggroup.args) atIndex:1];
            [enc setBuffer:gatebuf offset:(NSUInteger)gate_inner atIndex:2];
            [enc setBuffer:upbuf offset:(NSUInteger)up_inner atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.gate_offsets)
                    offset:ds4_gpu_tensor_offset(ggroup.gate_offsets) atIndex:4];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.up_offsets)
                    offset:ds4_gpu_tensor_offset(ggroup.up_offsets) atIndex:5];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:6];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:7];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.counts)
                    offset:ds4_gpu_tensor_offset(ggroup.counts) atIndex:8];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.pairs)
                    offset:ds4_gpu_tensor_offset(ggroup.pairs) atIndex:9];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:10];
            [enc useResource:gatebuf usage:MTLResourceUsageRead];
            [enc useResource:upbuf usage:MTLResourceUsageRead];
            [enc setThreadgroupMemoryLength:2u * (NSUInteger)expert_in_dim * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(expert_mid_dim, n_total_expert, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc setComputePipelineState:pair_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:gatebuf offset:(NSUInteger)gate_inner atIndex:1];
            [enc setBuffer:upbuf offset:(NSUInteger)up_inner atIndex:2];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
            [enc useResource:gatebuf usage:MTLResourceUsageRead];
            [enc useResource:upbuf usage:MTLResourceUsageRead];
            [enc setThreadgroupMemoryLength:512u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(expert_mid_dim, n_expert, n_tokens)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);
        if (gemm) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:gemm_down_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBytes:&ggroup.args length:sizeof(ggroup.args) atIndex:1];
            [enc setBuffer:downbuf offset:(NSUInteger)down_inner atIndex:2];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.down_offsets)
                    offset:ds4_gpu_tensor_offset(ggroup.down_offsets) atIndex:3];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:4];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.counts)
                    offset:ds4_gpu_tensor_offset(ggroup.counts) atIndex:5];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.pairs)
                    offset:ds4_gpu_tensor_offset(ggroup.pairs) atIndex:6];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.partial)
                    offset:ds4_gpu_tensor_offset(ggroup.partial) atIndex:7];
            [enc useResource:downbuf usage:MTLResourceUsageRead];
            [enc setThreadgroupMemoryLength:(NSUInteger)gemm_rows *
                 (NSUInteger)expert_mid_dim * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((out_dim + gemm_rows - 1u) / gemm_rows,
                                                  n_total_expert, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:gemm_reduce_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(ggroup.partial)
                    offset:ds4_gpu_tensor_offset(ggroup.partial) atIndex:1];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            const NSUInteger n_out = (NSUInteger)n_tokens * out_dim;
            [enc dispatchThreadgroups:MTLSizeMake((n_out + 255u) / 256u, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
            if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum Q8_0 expert-major MoE"))
                return 0;
            return 1;
        }
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:downbuf offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc useResource:downbuf usage:MTLResourceUsageRead];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_expert * 256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "Mellum Q8_0 batch routed MoE"))
            return 0;
    }
    return 1;
}
