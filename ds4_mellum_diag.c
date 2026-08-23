/*
 * Mellum 2's diagnostic harness: the twelve probe and profile entry points
 * behind `--mellum-diag`.  They are the checks that need a loaded model, as
 * distinct from the numeric kernel checks, which live in ds4_test.
 *
 * #included by ds4.c rather than compiled separately.  Each one drives a real
 * engine and session, so they reach struct ds4_engine and struct ds4_session,
 * which are opaque in ds4.h and defined only in ds4.c, plus the Mellum runtime
 * helpers beside them.  Only these twelve moved: the runtime itself is woven
 * through ds4.c's preprocessor regions -- several sit inside #ifndef
 * DS4_NO_GPU blocks thousands of lines long that they share with other model
 * families -- and cutting those apart mechanically is not safe.  These twelve
 * are each self-contained, conditionals included, which is why they can move
 * and the rest cannot yet.
 *
 * ds4.h already declares them, so no forward declarations are needed.
 */

int ds4_engine_mellum_layer0_probe(ds4_engine  *e,
                                   FILE        *out,
                                   const char  *raw_output_path) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)raw_output_path;
    fprintf(stderr, "ds4: Mellum layer-0 probe requires Metal support\n");
    return 1;
#else
    /* These are the exact rendered `python_add` ChatML fixture IDs recorded
     * from the pinned Mellum Q8 GGUF. Keep this probe session-free and fixed:
     * it is a numerical checkpoint tool, not a partial generation path. */
    static const int fixture_tokens[] = {
        27, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    const uint32_t n_tokens = (uint32_t)(sizeof(fixture_tokens) /
                                          sizeof(fixture_tokens[0]));
    if (!e || !out || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        !e->mellum_decode_contract_ready || e->backend != DS4_BACKEND_METAL ||
        !e->weights.token_embd) {
        fprintf(stderr, "ds4: Mellum layer-0 probe requires an inspect-loaded Metal Mellum engine\n");
        return 1;
    }

    ds4_gpu_mellum_q8_0_layer_desc desc;
    if (e->weights.token_embd->type != DS4_TENSOR_Q8_0 ||
        !ds4_mellum_q8_layer_desc(e, 0u, &desc)) {
        fprintf(stderr, "ds4: Mellum layer-0 probe requires Q8_0 dense/expert tensors and an F32 router\n");
        return 1;
    }

    const uint32_t n_embd = DS4_N_EMBD;
    const uint32_t q_dim = DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t kv_dim = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t q_bytes = (uint64_t)q_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)kv_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)n_tokens * kv_dim * sizeof(uint16_t);
    const uint64_t mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);

    ds4_gpu_tensor *hidden = NULL, *layer_out = NULL, *attention_out = NULL;
    ds4_gpu_tensor *attention_norm = NULL, *q = NULL, *k = NULL, *v = NULL;
    ds4_gpu_tensor *heads = NULL, *projected = NULL, *key_cache = NULL;
    ds4_gpu_tensor *value_cache = NULL, *ffn_norm = NULL, *router_logits = NULL;
    ds4_gpu_tensor *router_selected = NULL, *router_weights = NULL;
    ds4_gpu_tensor *router_probs = NULL, *moe_mid = NULL, *moe_out = NULL;
    float *hidden_cpu = NULL, *layer_cpu = NULL;
    FILE *raw_output = NULL;
    char *raw_output_tmp_path = NULL;
    uint64_t raw_output_values = 0;
    int32_t selected_cpu[8] = {0};
    float weights_cpu[8] = {0};
    int ok = ds4_gpu_set_model_map(e->model.map, e->model.size) != 0;
#define DS4_MELLUM_PROBE_ALLOC(name, bytes) \
    do { \
        (name) = ds4_gpu_tensor_alloc((bytes)); \
        if (!(name)) ok = 0; \
    } while (0)
    DS4_MELLUM_PROBE_ALLOC(hidden, embd_bytes);
    DS4_MELLUM_PROBE_ALLOC(layer_out, embd_bytes);
    DS4_MELLUM_PROBE_ALLOC(attention_out, embd_bytes);
    DS4_MELLUM_PROBE_ALLOC(attention_norm, embd_bytes);
    DS4_MELLUM_PROBE_ALLOC(q, q_bytes);
    DS4_MELLUM_PROBE_ALLOC(k, kv_bytes);
    DS4_MELLUM_PROBE_ALLOC(v, kv_bytes);
    DS4_MELLUM_PROBE_ALLOC(heads, q_bytes);
    DS4_MELLUM_PROBE_ALLOC(projected, embd_bytes);
    DS4_MELLUM_PROBE_ALLOC(key_cache, cache_bytes);
    DS4_MELLUM_PROBE_ALLOC(value_cache, cache_bytes);
    DS4_MELLUM_PROBE_ALLOC(ffn_norm, embd_bytes);
    DS4_MELLUM_PROBE_ALLOC(router_logits, DS4_N_EXPERT * sizeof(float));
    DS4_MELLUM_PROBE_ALLOC(router_selected, DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_MELLUM_PROBE_ALLOC(router_weights, DS4_N_EXPERT_USED * sizeof(float));
    DS4_MELLUM_PROBE_ALLOC(router_probs, DS4_N_EXPERT * sizeof(float));
    DS4_MELLUM_PROBE_ALLOC(moe_mid, mid_bytes);
    DS4_MELLUM_PROBE_ALLOC(moe_out, embd_bytes);
#undef DS4_MELLUM_PROBE_ALLOC
    hidden_cpu = xmalloc((size_t)embd_bytes);
    layer_cpu = xmalloc((size_t)embd_bytes);
    if (raw_output_path && raw_output_path[0]) {
        const size_t path_len = strlen(raw_output_path);
        static const char suffix[] = ".tmp.XXXXXX";
        raw_output_tmp_path = xmalloc(path_len + sizeof(suffix));
        snprintf(raw_output_tmp_path, path_len + sizeof(suffix), "%s%s",
                 raw_output_path, suffix);
        const int raw_output_fd = mkstemp(raw_output_tmp_path);
        if (raw_output_fd < 0 ||
            !(raw_output = fdopen(raw_output_fd, "wb"))) {
            const int saved_errno = errno;
            if (raw_output_fd >= 0) close(raw_output_fd);
            unlink(raw_output_tmp_path);
            fprintf(stderr, "ds4: could not stage Mellum layer-0 raw output: %s\n",
                    strerror(saved_errno));
            ok = 0;
        }
    }
    if (!ok) fprintf(stderr, "ds4: Mellum layer-0 probe allocation failed\n");

    /* Match llama.cpp's debug callback, which accumulates this checkpoint in
     * F32 rather than a diagnostic-only widened accumulator. */
    float layer_sum = 0.0f;
    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        embed_token_any(&e->model, &e->weights, fixture_tokens[pos], hidden_cpu);
        ok = ds4_gpu_tensor_write(hidden, 0, hidden_cpu, embd_bytes) != 0 &&
             ds4_gpu_mellum_q8_0_layer_decode_tensor(
                 layer_out, attention_out, attention_norm, q, k, v, heads,
                 projected, key_cache, value_cache, ffn_norm, router_logits,
                 router_selected, router_weights, router_probs, moe_mid, moe_out,
                 e->model.map, e->model.size, &desc, hidden, pos, n_tokens, 0u,
                 pos + 1u) != 0 &&
             ds4_gpu_tensor_read(layer_out, 0, layer_cpu, embd_bytes) != 0;
        if (ok && pos + 1u == n_tokens) {
            ok = ds4_gpu_tensor_read(router_selected, 0, selected_cpu,
                                     sizeof(selected_cpu)) != 0 &&
                 ds4_gpu_tensor_read(router_weights, 0, weights_cpu,
                                     sizeof(weights_cpu)) != 0;
        }
        if (ok && raw_output &&
            fwrite(layer_cpu, sizeof(float), n_embd, raw_output) != n_embd) {
            fprintf(stderr, "ds4: failed writing Mellum layer-0 raw output\n");
            ok = 0;
        }
        if (ok && raw_output) raw_output_values += n_embd;
        for (uint32_t i = 0; ok && i < n_embd; i++) layer_sum += layer_cpu[i];
    }

    if (raw_output && fclose(raw_output) != 0) ok = 0;
    raw_output = NULL;
    if (ok && raw_output_tmp_path &&
        raw_output_values != (uint64_t)n_tokens * n_embd) {
        fprintf(stderr, "ds4: Mellum layer-0 raw output has an unexpected length\n");
        ok = 0;
    }
    if (ok && raw_output_tmp_path &&
        rename(raw_output_tmp_path, raw_output_path) != 0) {
        fprintf(stderr, "ds4: could not finalize Mellum layer-0 raw output: %s\n",
                strerror(errno));
        ok = 0;
    }
    if (!ok && raw_output_tmp_path) unlink(raw_output_tmp_path);

    if (!ok) {
        fprintf(stderr, "ds4: Mellum layer-0 probe execution failed\n");
    } else {
        fprintf(out, "Mellum layer-0 probe tokens=%u sum=%.6f\n",
                n_tokens, layer_sum);
        fprintf(out, "Mellum layer-0 probe last=[%.7f, %.7f, %.7f, ..., %.7f, %.7f, %.7f]\n",
                layer_cpu[0], layer_cpu[1], layer_cpu[2],
                layer_cpu[n_embd - 3u], layer_cpu[n_embd - 2u],
                layer_cpu[n_embd - 1u]);
        fprintf(out, "Mellum layer-0 probe router=");
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            fprintf(out, "%s%d:%.7f", i ? "," : "", selected_cpu[i], weights_cpu[i]);
        }
        fputc('\n', out);
    }

    free(layer_cpu);
    free(hidden_cpu);
    ds4_gpu_tensor_free(moe_out); ds4_gpu_tensor_free(moe_mid);
    ds4_gpu_tensor_free(router_probs); ds4_gpu_tensor_free(router_weights);
    ds4_gpu_tensor_free(router_selected); ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(ffn_norm); ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache); ds4_gpu_tensor_free(projected);
    ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(attention_norm);
    ds4_gpu_tensor_free(attention_out); ds4_gpu_tensor_free(layer_out);
    ds4_gpu_tensor_free(hidden);
    free(raw_output_tmp_path);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_all_layers_probe(ds4_engine *e,
                                       FILE       *out,
                                       const char *raw_output_path,
                                       const char *trace_output_path,
                                       const char *attention_trace_output_path,
                                       const char *qk_trace_output_path) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)raw_output_path;
    (void)trace_output_path;
    (void)attention_trace_output_path;
    (void)qk_trace_output_path;
    fprintf(stderr, "ds4: Mellum all-layer probe requires Metal support\n");
    return 1;
#else
    /* Keep the first whole-model pass as a fixed oracle diagnostic. It reuses
     * private engine-owned scratch/KV state, but creates no session, enables
     * no normal Mellum execution, and never reaches the shared KV store. */
    static const int fixture_tokens[] = {
        27, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    const uint32_t n_tokens = (uint32_t)(sizeof(fixture_tokens) /
                                          sizeof(fixture_tokens[0]));
    if (!e || !out || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        !e->mellum_decode_contract_ready || e->backend != DS4_BACKEND_METAL ||
        !e->weights.token_embd || e->weights.token_embd->type != DS4_TENSOR_Q8_0) {
        fprintf(stderr, "ds4: Mellum all-layer probe requires an inspect-loaded Q8 Metal Mellum engine\n");
        return 1;
    }

    const uint32_t n_embd = DS4_N_EMBD;
    const uint32_t q_dim = DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t kv_dim = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    float *final_cpu = xmalloc((size_t)embd_bytes);
    float *layer_trace = xmalloc((size_t)DS4_N_LAYER * embd_bytes);
    float *attention_trace = xmalloc((size_t)DS4_N_LAYER * embd_bytes);
    const uint32_t qk_trace_stride = q_dim + kv_dim;
    float *qk_trace = xmalloc((size_t)DS4_N_LAYER * qk_trace_stride * sizeof(float));
    int ok = ds4_engine_mellum_decode_state_prepare(e, n_tokens);
    if (ok) ds4_mellum_decode_state_reset(e->mellum_decode_state);

    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        const bool capture_trace = pos + 1u == n_tokens;
        ok = ds4_mellum_decode_token(
            e, e->mellum_decode_state, fixture_tokens[pos],
            capture_trace ? final_cpu : NULL,
            capture_trace ? layer_trace : NULL,
            capture_trace ? attention_trace : NULL,
            capture_trace ? qk_trace : NULL, NULL, NULL, NULL, false, false);
        if (!ok) {
            fprintf(stderr, "ds4: Mellum inspect graph failed at token %u\n", pos);
        }
    }
    if (ok && raw_output_path && raw_output_path[0] &&
        !ds4_mellum_write_atomic(raw_output_path, final_cpu, embd_bytes)) {
        fprintf(stderr, "ds4: could not write Mellum all-layer raw output\n");
        ok = 0;
    }
    if (ok && trace_output_path && trace_output_path[0] &&
        !ds4_mellum_write_atomic(trace_output_path, layer_trace,
                                 (uint64_t)DS4_N_LAYER * embd_bytes)) {
        fprintf(stderr, "ds4: could not write Mellum all-layer trace output\n");
        ok = 0;
    }
    if (ok && attention_trace_output_path && attention_trace_output_path[0] &&
        !ds4_mellum_write_atomic(attention_trace_output_path, attention_trace,
                                 (uint64_t)DS4_N_LAYER * embd_bytes)) {
        fprintf(stderr, "ds4: could not write Mellum all-layer attention trace output\n");
        ok = 0;
    }
    if (ok && qk_trace_output_path && qk_trace_output_path[0] &&
        !ds4_mellum_write_atomic(qk_trace_output_path, qk_trace,
                                 (uint64_t)DS4_N_LAYER * qk_trace_stride * sizeof(float))) {
        fprintf(stderr, "ds4: could not write Mellum all-layer Q/K trace output\n");
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "ds4: Mellum all-layer probe execution failed\n");
    } else {
        float final_sum = 0.0f;
        for (uint32_t i = 0; i < n_embd; i++) final_sum += final_cpu[i];
        fprintf(out, "Mellum all-layer probe tokens=%u final-sum=%.6f\n", n_tokens, final_sum);
        fprintf(out, "Mellum all-layer probe last=[%.7f, %.7f, %.7f, ..., %.7f, %.7f, %.7f]\n",
                final_cpu[0], final_cpu[1], final_cpu[2],
                final_cpu[n_embd - 3u], final_cpu[n_embd - 2u],
                final_cpu[n_embd - 1u]);
    }

    free(final_cpu);
    free(layer_trace);
    free(attention_trace);
    free(qk_trace);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_kv_layout_probe(ds4_engine *e,
                                      FILE       *out,
                                      int         ctx_size) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)ctx_size;
    fprintf(stderr, "ds4: Mellum KV-layout probe requires Metal support\n");
    return 1;
#else
    if (!e || !out || ctx_size <= 0 ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        !e->mellum_decode_contract_ready || e->backend != DS4_BACKEND_METAL) {
        fprintf(stderr, "ds4: Mellum KV-layout probe requires an inspect-loaded Metal Mellum engine\n");
        return 1;
    }
    ds4_mellum_kv_layout layout = {0};
    uint64_t total_bytes = 0;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint32_t full_layers = DS4_N_LAYER / 4u;
    const uint32_t sliding_layers = DS4_N_LAYER - full_layers;
    const bool ok = ds4_mellum_kv_layout_alloc(
        e, &layout, (uint32_t)ctx_size, &total_bytes);
    if (!ok) {
        fprintf(stderr, "ds4: Mellum KV-layout probe allocation failed\n");
    } else {
        fprintf(out,
                "Mellum KV-layout probe ctx=%d layers=%u sliding=%u cap=%u full=%u cap=%d kv-dim=%llu f16-bytes=%llu\n",
                ctx_size, DS4_N_LAYER, sliding_layers, DS4_N_SWA, full_layers,
                ctx_size, (unsigned long long)kv_dim,
                (unsigned long long)total_bytes);
        fprintf(out,
                "Mellum KV-layout probe allocated then released private F16 tensors; no session or token evaluation occurred\n");
    }
    ds4_mellum_kv_layout_free(&layout);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_logits_probe(ds4_engine *e,
                                   FILE       *out,
                                   const char *raw_output_path,
                                   uint32_t    report_top_k) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)raw_output_path;
    (void)report_top_k;
    fprintf(stderr, "ds4: Mellum logits probe requires Metal support\n");
    return 1;
#else
    /* This deliberately stops at raw logits. It is a fixed numerical seam,
     * not an implicit authorization to sample, emit, or generate a token. */
    static const int fixture_tokens[] = {
        27, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    const uint32_t n_tokens = (uint32_t)(sizeof(fixture_tokens) /
                                          sizeof(fixture_tokens[0]));
    if (!e || !out || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        !e->mellum_decode_contract_ready || e->backend != DS4_BACKEND_METAL ||
        !e->weights.token_embd || e->weights.token_embd->type != DS4_TENSOR_Q8_0 ||
        !e->weights.output || e->weights.output->dim[1] > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "ds4: Mellum logits probe requires an inspect-loaded Q8 Metal Mellum engine\n");
        return 1;
    }
    const uint64_t vocab_dim = e->weights.output->dim[1];
    float *logits_cpu = xmalloc((size_t)vocab_dim * sizeof(float));
    int ok = ds4_engine_mellum_decode_state_prepare(e, n_tokens);
    if (ok) ok = ds4_mellum_decode_output_prepare(e, e->mellum_decode_state);
    if (ok) ds4_mellum_decode_state_reset(e->mellum_decode_state);
    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        ok = ds4_mellum_decode_token(
            e, e->mellum_decode_state, fixture_tokens[pos],
            NULL, NULL, NULL, NULL,
            NULL, NULL, pos + 1u == n_tokens ? logits_cpu : NULL, false, false);
        if (!ok) fprintf(stderr, "ds4: Mellum logits probe failed at token %u\n", pos);
    }
    if (ok && raw_output_path && raw_output_path[0] &&
        !ds4_mellum_write_atomic(raw_output_path, logits_cpu,
                                 vocab_dim * sizeof(float))) {
        fprintf(stderr, "ds4: could not write Mellum logits probe output\n");
        ok = 0;
    }
    if (ok && report_top_k) {
        ok = ds4_mellum_print_logits_top_k(out, logits_cpu, vocab_dim,
                                           report_top_k);
    }
    if (ok) {
        float sum = 0.0f;
        for (uint64_t i = 0; i < vocab_dim; i++) sum += logits_cpu[i];
        fprintf(out, "Mellum logits probe tokens=%u vocab=%llu sum=%.6f\n",
                n_tokens, (unsigned long long)vocab_dim, sum);
        fprintf(out, "Mellum logits probe raw=[%.7f, %.7f, %.7f, ..., %.7f, %.7f, %.7f]\n",
                logits_cpu[0], logits_cpu[1], logits_cpu[2],
                logits_cpu[vocab_dim - 3u], logits_cpu[vocab_dim - 2u],
                logits_cpu[vocab_dim - 1u]);
    } else {
        fprintf(stderr, "ds4: Mellum logits probe execution failed\n");
    }
    free(logits_cpu);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_session_lifecycle_probe(ds4_engine *e,
                                              FILE       *out,
                                              int         ctx_size) {
    if (!e || !out || ctx_size <= 0 ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready) {
        fprintf(stderr, "ds4: Mellum session lifecycle probe requires an inspect-loaded Metal engine\n");
        return 1;
    }
    ds4_session *session = NULL;
    if (ds4_mellum_session_create(&session, e, ctx_size, false, false) != 0 || !session ||
        !ds4_session_is_mellum_layout_only(session)) {
        fprintf(stderr, "ds4: Mellum layout-only session creation failed\n");
        ds4_session_free(session);
        return 1;
    }
    char reject_err[128] = "";
    if (ds4_session_eval(session, 0, reject_err, sizeof(reject_err)) == 0 ||
        strstr(reject_err, "layout-only") == NULL ||
        ds4_session_eval_argmax(session, 0, reject_err, sizeof(reject_err)) != -1 ||
        ds4_session_argmax(session) != -1 ||
        ds4_session_argmax_excluding(session, 0) != -1 ||
        ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, NULL) != -1) {
        fprintf(stderr, "ds4: Mellum layout-only session unexpectedly enabled token execution or selection\n");
        ds4_session_free(session);
        return 1;
    }
    ds4_token_score score = {0};
    if (ds4_session_top_logprobs(session, &score, 1) != 0 ||
        ds4_session_token_logprob(session, 0, &score) != 0) {
        fprintf(stderr, "ds4: Mellum layout-only session unexpectedly exposed logits\n");
        ds4_session_free(session);
        return 1;
    }
    ds4_session_payload_file staged = {0};
    ds4_session_snapshot snapshot = {0};
    if (ds4_session_payload_bytes(session) != 0 ||
        ds4_session_layer_payload_bytes(session, 0, 0) != 0 ||
        ds4_session_stage_payload(session, &staged, reject_err,
                                  sizeof(reject_err)) != 1 ||
        ds4_session_save_payload(session, NULL, reject_err,
                                 sizeof(reject_err)) != 1 ||
        ds4_session_load_payload(session, NULL, 0, reject_err,
                                 sizeof(reject_err)) != 1 ||
        ds4_session_save_layer_payload(session, NULL, 0, 0, reject_err,
                                       sizeof(reject_err)) != 1 ||
        ds4_session_load_layer_payload(session, NULL, 0, NULL, 0, 0, 0,
                                       reject_err, sizeof(reject_err)) != 1 ||
        ds4_session_save_snapshot(session, &snapshot, reject_err,
                                  sizeof(reject_err)) != 1 ||
        ds4_session_load_snapshot(session, NULL, reject_err,
                                  sizeof(reject_err)) != 1) {
        fprintf(stderr, "ds4: Mellum layout-only session unexpectedly exposed payload state\n");
        ds4_session_payload_file_free(&staged);
        ds4_session_snapshot_free(&snapshot);
        ds4_session_free(session);
        return 1;
    }
    const uint32_t full_layers = DS4_N_LAYER / 4u;
    const uint32_t sliding_layers = DS4_N_LAYER - full_layers;
    fprintf(out,
            "Mellum session lifecycle probe created and released layout-only session ctx=%d sliding=%u full=%u eval=rejected\n",
            ctx_size, sliding_layers, full_layers);
    ds4_session_free(session);
    return 0;
}

int ds4_engine_mellum_session_decode_probe(ds4_engine *e,
                                           FILE       *out,
                                           int         ctx_size,
                                           const char *raw_output_path) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)ctx_size;
    (void)raw_output_path;
    fprintf(stderr, "ds4: Mellum session decode probe requires Metal support\n");
    return 1;
#else
    static const int fixture_tokens[] = {
        27, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    const uint32_t n_tokens = (uint32_t)(sizeof(fixture_tokens) /
                                          sizeof(fixture_tokens[0]));
    if (!e || !out || ctx_size < (int)n_tokens ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready) {
        fprintf(stderr, "ds4: Mellum session decode probe requires an inspect-loaded Metal engine\n");
        return 1;
    }
    ds4_session *session = NULL;
    if (ds4_mellum_session_create(&session, e, ctx_size, true, false) != 0 ||
        !session || !ds4_session_is_mellum_decode_only(session)) {
        fprintf(stderr, "ds4: Mellum inspect decode session creation failed\n");
        ds4_session_free(session);
        return 1;
    }
    char err[128] = "";
    int ok = 1;
    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        if (ds4_session_eval(session, fixture_tokens[pos], err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: Mellum session decode probe failed at token %u: %s\n",
                    pos, err[0] ? err : "unknown error");
            ok = 0;
        }
    }
    const uint64_t vocab_dim = e->weights.output->dim[1];
    float *logits = ok ? xmalloc((size_t)vocab_dim * sizeof(*logits)) : NULL;
    if (ok && (ds4_session_copy_logits(session, logits, (int)vocab_dim) !=
                   (int)vocab_dim ||
               ds4_session_argmax(session) != -1 ||
               ds4_session_eval_argmax(session, 0, err, sizeof(err)) != -1 ||
               ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, NULL) != -1 ||
               ds4_session_top_logprobs(session, &(ds4_token_score){0}, 1) != 0 ||
               ds4_session_payload_bytes(session) != 0 ||
               ds4_session_sync(session, NULL, err, sizeof(err)) != 1)) {
        fprintf(stderr, "ds4: Mellum session decode probe exposed an unsupported operation\n");
        ok = 0;
    }
    if (ok && raw_output_path && raw_output_path[0] &&
        !ds4_mellum_write_atomic(raw_output_path, logits,
                                 vocab_dim * sizeof(*logits))) {
        fprintf(stderr, "ds4: could not write Mellum session decode output\n");
        ok = 0;
    }
    if (ok) {
        float sum = 0.0f;
        for (uint64_t i = 0; i < vocab_dim; i++) sum += logits[i];
        fprintf(out,
                "Mellum session decode probe tokens=%u ctx=%d vocab=%llu sum=%.6f selection=rejected\n",
                n_tokens, ctx_size, (unsigned long long)vocab_dim, sum);
    }
    ds4_session_rewind(session, (int)n_tokens - 1);
    if (ok && ds4_session_pos(session) != 0) {
        fprintf(stderr, "ds4: Mellum inspect rewind did not reset session state\n");
        ok = 0;
    }
    free(logits);
    ds4_session_free(session);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_session_isolation_probe(ds4_engine *e,
                                              FILE       *out,
                                              int         ctx_size) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)ctx_size;
    fprintf(stderr, "ds4: Mellum session isolation probe requires Metal support\n");
    return 1;
#else
    static const int fixture_a[] = {
        27, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    static const int fixture_b[] = {
        28, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    const uint32_t n_tokens = (uint32_t)(sizeof(fixture_a) /
                                          sizeof(fixture_a[0]));
    if (!e || !out || ctx_size < (int)n_tokens ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready) {
        fprintf(stderr, "ds4: Mellum session isolation probe requires an inspect-loaded Metal engine\n");
        return 1;
    }
    char err[128] = "";
    const uint64_t vocab_dim = e->weights.output->dim[1];
    const size_t logit_bytes = (size_t)vocab_dim * sizeof(float);
    float *a_reference = xmalloc(logit_bytes);
    float *b_reference = xmalloc(logit_bytes);
    float *a_logits = xmalloc(logit_bytes);
    float *b_logits = xmalloc(logit_bytes);
    ds4_session *baseline = NULL, *a = NULL, *b = NULL;
    int ok = ds4_mellum_session_create(&baseline, e, ctx_size, true, false) == 0 &&
             ds4_mellum_session_decode_fixture(baseline, fixture_a, n_tokens,
                                               err, sizeof(err)) &&
             ds4_session_copy_logits(baseline, a_reference, (int)vocab_dim) ==
                 (int)vocab_dim;
    ds4_session_free(baseline);
    baseline = NULL;
    if (ok) {
        ok = ds4_mellum_session_create(&baseline, e, ctx_size, true, false) == 0 &&
             ds4_mellum_session_decode_fixture(baseline, fixture_b, n_tokens,
                                               err, sizeof(err)) &&
             ds4_session_copy_logits(baseline, b_reference, (int)vocab_dim) ==
                 (int)vocab_dim;
    }
    ds4_session_free(baseline);
    baseline = NULL;
    if (ok) {
        ok = ds4_mellum_session_create(&a, e, ctx_size, true, false) == 0 &&
             ds4_mellum_session_create(&b, e, ctx_size, true, false) == 0;
    }
    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        ok = ds4_session_eval(a, fixture_a[pos], err, sizeof(err)) == 0 &&
             ds4_session_eval(b, fixture_b[pos], err, sizeof(err)) == 0;
    }
    if (!ok) {
        fprintf(stderr, "ds4: Mellum session isolation decode failed: %s\n",
                err[0] ? err : "allocation or copy failure");
    }
    if (ok && (ds4_session_copy_logits(a, a_logits, (int)vocab_dim) !=
                   (int)vocab_dim ||
               ds4_session_copy_logits(b, b_logits, (int)vocab_dim) !=
                   (int)vocab_dim ||
               !ds4_mellum_logits_are_finite(a_reference, vocab_dim) ||
               !ds4_mellum_logits_are_finite(b_reference, vocab_dim) ||
               !ds4_mellum_logits_are_finite(a_logits, vocab_dim) ||
               !ds4_mellum_logits_are_finite(b_logits, vocab_dim) ||
               memcmp(a_reference, a_logits, logit_bytes) != 0 ||
               memcmp(b_reference, b_logits, logit_bytes) != 0)) {
        fprintf(stderr, "ds4: Mellum session isolation logits diverged or were non-finite\n");
        ok = 0;
    }
    if (ok && (ds4_session_argmax(a) != -1 || ds4_session_argmax(b) != -1)) {
        fprintf(stderr, "ds4: Mellum session isolation exposed token selection\n");
        ok = 0;
    }
    if (ok) {
        fprintf(out,
                "Mellum session isolation probe sessions=2 tokens=%u ctx=%d logits=f32-exact selection=rejected\n",
                n_tokens, ctx_size);
    }
    ds4_session_free(baseline);
    free(b_logits);
    free(a_logits);
    free(b_reference);
    free(a_reference);
    ds4_session_free(b);
    ds4_session_free(a);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_interactive_session_probe(ds4_engine *e,
                                                FILE       *out,
                                                int         ctx_size) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)ctx_size;
    fprintf(stderr, "ds4: Mellum interactive session probe requires Metal support\n");
    return 1;
#else
    static int fixture_a[] = {27, 1397, 233, 12998, 497, 2717};
    static int fixture_b[] = {28, 1397, 233, 12998, 497, 2717};
    const int n_tokens = (int)(sizeof(fixture_a) / sizeof(fixture_a[0]));
    if (!e || !out || ctx_size <= n_tokens ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready ||
        e->mellum_interactive_sessions) {
        fprintf(stderr,
                "ds4: Mellum interactive session probe requires an inspect-loaded Metal engine\n");
        return 1;
    }

    char err[160] = "";
    ds4_session *ordinary = NULL;
    ds4_session *baseline = NULL;
    ds4_session *session = NULL;
    const uint64_t vocab_dim = e->weights.output->dim[1];
    const size_t logit_bytes = (size_t)vocab_dim * sizeof(float);
    float *a_reference = xmalloc(logit_bytes);
    float *b_reference = xmalloc(logit_bytes);
    float *actual = xmalloc(logit_bytes);
    ds4_tokens a_prefix = {.v = fixture_a, .len = 3, .cap = 3};
    ds4_tokens a_full = {.v = fixture_a, .len = n_tokens, .cap = n_tokens};
    ds4_tokens b_full = {.v = fixture_b, .len = n_tokens, .cap = n_tokens};

    int ok = ds4_session_create(&ordinary, e, ctx_size) == 0 && ordinary &&
             ds4_session_is_mellum_layout_only(ordinary) &&
             !ds4_session_is_mellum_interactive(ordinary) &&
             ds4_session_argmax(ordinary) == -1;
    ds4_session_free(ordinary);
    ordinary = NULL;

    if (ok) {
        ok = ds4_mellum_session_create(
                 &baseline, e, ctx_size, true, true) == 0 && baseline &&
             ds4_session_sync(baseline, &b_full, err, sizeof(err)) == 0 &&
             ds4_session_copy_logits(baseline, b_reference, (int)vocab_dim) ==
                 (int)vocab_dim;
    }
    ds4_session_free(baseline);
    baseline = NULL;

    if (ok) {
        ok = ds4_mellum_session_create(
                 &session, e, ctx_size, true, true) == 0 && session &&
             ds4_session_is_mellum_interactive(session) &&
             !ds4_session_supports_payload(session) &&
             ds4_session_sync(session, &a_prefix, err, sizeof(err)) == 0 &&
             ds4_session_pos(session) == a_prefix.len &&
             ds4_session_sync(session, &a_full, err, sizeof(err)) == 0 &&
             ds4_session_pos(session) == a_full.len &&
             ds4_session_common_prefix(session, &a_full) == a_full.len &&
             ds4_session_copy_logits(session, a_reference, (int)vocab_dim) ==
                 (int)vocab_dim;
    }

    uint64_t rng = 1;
    ds4_token_score scores[2] = {0};
    const int argmax = ok ? ds4_session_argmax(session) : -1;
    if (ok && (argmax < 0 ||
               ds4_session_argmax_excluding(session, argmax) < 0 ||
               ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng) < 0 ||
               ds4_session_top_logprobs(session, scores, 2) != 2 ||
               ds4_session_token_logprob(session, argmax, &scores[0]) != 1)) {
        fprintf(stderr, "ds4: Mellum interactive session did not expose token selection\n");
        ok = 0;
    }

    /* A Mellum token owns its command batch. It must refuse rather than close
     * a batch opened by a caller, then remain safe to rebuild after that
     * failed attempt. This regression locks the ownership guard in the fast
     * GPU-resident decode path. */
    if (ok) {
        const bool caller_batch = ds4_gpu_begin_commands() != 0;
        const int eval_rc = caller_batch ?
            ds4_session_eval(session, fixture_a[0], err, sizeof(err)) : 0;
        const bool caller_batch_preserved =
            caller_batch && eval_rc != 0 && ds4_gpu_commands_active() &&
            ds4_session_pos(session) == 0 && ds4_gpu_end_commands() != 0;
        if (!caller_batch_preserved ||
            ds4_session_sync(session, &a_full, err, sizeof(err)) != 0 ||
            ds4_session_copy_logits(session, actual, (int)vocab_dim) !=
                (int)vocab_dim ||
            memcmp(actual, a_reference, logit_bytes) != 0) {
            fprintf(stderr,
                    "ds4: Mellum caller-owned command batch was not preserved\n");
            ok = 0;
        }
    }

    if (ok &&
        (ds4_session_sync(session, &b_full, err, sizeof(err)) != 0 ||
         ds4_session_pos(session) != b_full.len ||
         ds4_session_copy_logits(session, actual, (int)vocab_dim) !=
             (int)vocab_dim ||
         memcmp(actual, b_reference, logit_bytes) != 0)) {
        fprintf(stderr,
                "ds4: Mellum divergent prompt replay did not reproduce its baseline\n");
        ok = 0;
    }

    ds4_mellum_sync_cancel_probe cancel_probe = {
        .cancel_after = 2,
    };
    if (ok) {
        ds4_session_invalidate(session);
        ds4_session_set_progress(session, ds4_mellum_sync_cancel_progress,
                                 &cancel_probe);
        ds4_session_set_cancel(session, ds4_mellum_sync_cancelled,
                               &cancel_probe);
        const int sync_rc = ds4_session_sync(session, &a_full, err, sizeof(err));
        ds4_session_set_cancel(session, NULL, NULL);
        ds4_session_set_progress(session, NULL, NULL);
        if (sync_rc != DS4_SESSION_SYNC_INTERRUPTED ||
            ds4_session_pos(session) != cancel_probe.cancel_after ||
            ds4_session_common_prefix(session, &a_full) !=
                cancel_probe.cancel_after ||
            ds4_session_sync(session, &a_full, err, sizeof(err)) != 0 ||
            ds4_session_copy_logits(session, actual, (int)vocab_dim) !=
                (int)vocab_dim ||
            memcmp(actual, a_reference, logit_bytes) != 0) {
            fprintf(stderr,
                    "ds4: Mellum interrupted prompt replay did not resume exactly\n");
            ok = 0;
        }
    }

    if (ok) {
        ds4_session_rewind(session, 3);
        if (ds4_session_pos(session) != 0 ||
            ds4_session_sync(session, &a_full, err, sizeof(err)) != 0 ||
            ds4_session_copy_logits(session, actual, (int)vocab_dim) !=
                (int)vocab_dim ||
            memcmp(actual, a_reference, logit_bytes) != 0) {
            fprintf(stderr, "ds4: Mellum rewind/reset replay was not exact\n");
            ok = 0;
        }
    }

    if (!ok && err[0]) fprintf(stderr, "ds4: Mellum interactive probe: %s\n", err);
    if (ok) {
        fprintf(out,
                "Mellum interactive session probe tokens=%d ctx=%d sync=fresh,extend,rebuild,resume reset=exact selection=enabled batch=caller-preserved inspect=blocked payload=transcript-only\n",
                n_tokens, ctx_size);
    }
    ds4_session_free(session);
    ds4_session_free(baseline);
    ds4_session_free(ordinary);
    free(actual);
    free(b_reference);
    free(a_reference);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_swa_boundary_probe(ds4_engine *e,
                                         FILE       *out,
                                         int         ctx_size) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)ctx_size;
    fprintf(stderr, "ds4: Mellum SWA boundary probe requires Metal support\n");
    return 1;
#else
    enum { n_tokens = 1030 };
    if (!e || !out || ctx_size <= n_tokens ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready ||
        e->mellum_interactive_sessions) {
        fprintf(stderr,
                "ds4: Mellum SWA boundary probe requires an inspect-loaded Metal engine and ctx > %d\n",
                n_tokens);
        return 1;
    }
    int *tokens = xmalloc((size_t)n_tokens * sizeof(*tokens));
    for (int i = 0; i < n_tokens; i++) {
        /* Valid, varied IDs; the deterministic sequence avoids tokenizer state. */
        tokens[i] = (int)(((uint32_t)i * 7919u + 27u) % DS4_N_VOCAB);
    }
    ds4_tokens prompt = {.v = tokens, .len = n_tokens, .cap = n_tokens};
    const uint64_t vocab_dim = e->weights.output->dim[1];
    const size_t logit_bytes = (size_t)vocab_dim * sizeof(float);
    float *reference = xmalloc(logit_bytes);
    float *actual = xmalloc(logit_bytes);
    ds4_session *baseline = NULL;
    ds4_session *batched = NULL;
    char err[160] = "";
    bool ok = ds4_mellum_session_create(&baseline, e, ctx_size, true, true) == 0 &&
              baseline != NULL &&
              ds4_mellum_session_decode_fixture(baseline, tokens, n_tokens,
                                                err, sizeof(err)) &&
              ds4_session_copy_logits(baseline, reference, (int)vocab_dim) ==
                  (int)vocab_dim;
    if (ok) {
        ok = ds4_mellum_session_create(&batched, e, ctx_size, true, true) == 0 &&
             batched != NULL &&
             ds4_session_sync(batched, &prompt, err, sizeof(err)) == 0 &&
             ds4_session_pos(batched) == n_tokens &&
             ds4_session_copy_logits(batched, actual, (int)vocab_dim) ==
                 (int)vocab_dim &&
             ds4_mellum_logits_are_finite(reference, vocab_dim) &&
             ds4_mellum_logits_are_finite(actual, vocab_dim);
    }
    /*
     * A 1,030-token history is past the head-grouping threshold, so decode
     * reassociates here however the engine is configured and this can no
     * longer be a bitwise comparison.  Bound it instead: measured deviation is
     * ~0.1 against a logit scale of ~22, and a limit two orders above that
     * still catches a broken merge -- zeroed output, dropped stripes, a
     * corrupted running maximum -- all of which move the result by whole
     * units.  Bitwise coverage lives where it can: the true-prefill probe and
     * ds4_test's batch-versus-decode check against the oracle configuration.
     */
    double max_abs = 0.0, sum_sq = 0.0;
    if (ok) {
        for (uint64_t i = 0; i < vocab_dim; i++) {
            const double d = (double)actual[i] - (double)reference[i];
            if (fabs(d) > max_abs) max_abs = fabs(d);
            sum_sq += d * d;
        }
        const double max_abs_limit = 0.5;
        ok = max_abs <= max_abs_limit;
        if (!ok) {
            snprintf(err, sizeof(err),
                     "logits deviate max_abs=%g beyond %g",
                     max_abs, max_abs_limit);
        }
    }
    (void)logit_bytes;
    if (!ok) {
        fprintf(stderr, "ds4: Mellum SWA boundary schedules diverged: %s\n",
                err[0] ? err : "allocation, execution, or F32 comparison failure");
    } else {
        fprintf(out,
                "Mellum SWA boundary probe tokens=%d boundary=1024 sliding=21 full=7 ctx=%d logits max_abs=%g rms=%g\n",
                n_tokens, ctx_size, max_abs, sqrt(sum_sq / (double)vocab_dim));
    }
    ds4_session_free(batched);
    ds4_session_free(baseline);
    free(actual);
    free(reference);
    free(tokens);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_resident_profile(ds4_engine *e, FILE *out,
                                       int ctx_size) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    (void)ctx_size;
    fprintf(stderr, "ds4: Mellum resident profile requires Metal support\n");
    return 1;
#else
    enum { warmup_tokens = 8, measured_tokens = 64, repeats = 3 };
    /*
     * Prefill width is a diagnostic knob rather than a constant so the
     * throughput-versus-context curve can be measured: prefilling N tokens from
     * an empty cache and differencing the cumulative times across N gives the
     * marginal rate at depth, which is the number a long session actually
     * experiences.  Defaults to the sliding-window cap.
     */
    const ds4_mellum_runtime *rt = ds4_mellum_runtime_get();
    int prefill_tokens = rt->profile_prefill_tokens ?
        rt->profile_prefill_tokens : 1024;
    /*
     * Decode depth: prime the cache with this many tokens before timing the
     * decode passes.  Zero keeps the historical empty-cache measurement.
     */
    const int decode_depth = rt->profile_decode_depth;
    ds4_mellum_prefill_scratch depth_scratch = {0};
    int     *depth_toks  = NULL;
    uint32_t depth_n     = 0;
    uint32_t depth_chunk = 0;
    if (!e || !out || ctx_size <= measured_tokens ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready ||
        e->mellum_interactive_sessions) {
        fprintf(stderr, "ds4: Mellum resident profile requires an inspect-loaded Metal engine\n");
        return 1;
    }
    int tokens[measured_tokens];
    for (int i = 0; i < measured_tokens; i++) {
        tokens[i] = (int)(((uint32_t)i * 7919u + 27u) % DS4_N_VOCAB);
    }
    ds4_session *session = NULL;
    bool ok = ds4_mellum_session_create(&session, e, ctx_size, true, true) == 0 &&
              session && session->mellum && session->mellum->decode;
    double layers_sec = 0.0, logits_sec = 0.0, pass_sec = 0.0;
    if (ok && decode_depth > 0) {
        if (ctx_size < decode_depth + measured_tokens) {
            fprintf(stderr,
                    "ds4: Mellum resident profile decode depth %d needs ctx >= %d\n",
                    decode_depth, decode_depth + measured_tokens);
            ok = false;
        } else {
            depth_n = (uint32_t)decode_depth;
            depth_chunk = ds4_mellum_probe_chunk(depth_n < 1024u ? depth_n : 1024u,
                                                 depth_n);
            depth_toks = xmalloc((size_t)depth_n * sizeof(*depth_toks));
            for (uint32_t i = 0; i < depth_n; i++) {
                depth_toks[i] = (int)((i * 7919u + 27u) % DS4_N_VOCAB);
            }
            ok = ds4_mellum_prefill_scratch_create(&depth_scratch, depth_chunk);
            if (!ok) fprintf(stderr, "ds4: Mellum decode-depth priming failed\n");
        }
    }
    if (ok) {
        /* Warm both paths before timing to avoid compilation/first-use costs. */
        ok = ds4_mellum_resident_profile_pass(e, session->mellum->decode,
                                              tokens, warmup_tokens, false,
                                              &pass_sec, &depth_scratch,
                                              depth_toks, depth_n, depth_chunk) &&
             ds4_mellum_resident_profile_pass(e, session->mellum->decode,
                                              tokens, warmup_tokens, true,
                                              &pass_sec, &depth_scratch,
                                              depth_toks, depth_n, depth_chunk);
    }
    for (int i = 0; ok && i < repeats; i++) {
        ok = ds4_mellum_resident_profile_pass(e, session->mellum->decode,
                                              tokens, measured_tokens, false,
                                              &pass_sec, &depth_scratch,
                                              depth_toks, depth_n, depth_chunk);
        layers_sec += pass_sec;
        ok = ok && ds4_mellum_resident_profile_pass(
                       e, session->mellum->decode, tokens, measured_tokens,
                       true, &pass_sec, &depth_scratch, depth_toks, depth_n,
                       depth_chunk);
        logits_sec += pass_sec;
    }
    /*
     * Layer-major prefill at the sliding-window cap.  This is the throughput
     * number sequential sync cannot reach; it is measured here so the
     * projection-precision choice can be judged against its real cost.
     */
    double prefill_sec = 0.0;
    bool prefill_ok = false;
    uint32_t prefill_chunk = prefill_tokens;
    ds4_mellum_prefill_scratch prefill_scratch = {0};
    int *prefill_toks = NULL;
    if (ok && ctx_size >= prefill_tokens) {
        prefill_toks = xmalloc((size_t)prefill_tokens * sizeof(*prefill_toks));
        for (int i = 0; i < prefill_tokens; i++) {
            prefill_toks[i] = (int)(((uint32_t)i * 7919u + 27u) % DS4_N_VOCAB);
        }
        prefill_chunk = ds4_mellum_probe_chunk(prefill_tokens, prefill_tokens);
        prefill_ok = ds4_mellum_prefill_scratch_create(&prefill_scratch,
                                                       prefill_chunk) &&
                     ds4_mellum_resident_prefill_pass(
                         e, session->mellum->decode, &prefill_scratch,
                         prefill_toks, prefill_tokens, prefill_chunk, &pass_sec);
        for (int i = 0; prefill_ok && i < repeats; i++) {
            prefill_ok = ds4_mellum_resident_prefill_pass(
                e, session->mellum->decode, &prefill_scratch, prefill_toks,
                prefill_tokens, prefill_chunk, &pass_sec);
            prefill_sec += pass_sec;
        }
    }
    if (ok) {
        const double no_head_ms = layers_sec * 1000.0 / (repeats * measured_tokens);
        const double logits_ms = logits_sec * 1000.0 / (repeats * measured_tokens);
        fprintf(out,
                "Mellum resident profile tokens=%d depth=%d repeats=%d no-head=%.3fms %.1ft/s with-head=%.3fms %.1ft/s output-head-delta=%.3fms\n",
                measured_tokens, decode_depth, repeats, no_head_ms, 1000.0 / no_head_ms,
                logits_ms, 1000.0 / logits_ms, logits_ms - no_head_ms);
        if (prefill_ok) {
            const double prefill_ms =
                prefill_sec * 1000.0 / (double)repeats;
            fprintf(out,
                    "Mellum resident prefill tokens=%d chunk=%u exact=%d repeats=%d batch=%.1fms %.1ft/s\n",
                    prefill_tokens, prefill_chunk,
                    ds4_gpu_mellum_prefill_exact_projections_enabled(),
                    repeats, prefill_ms,
                    (double)prefill_tokens * 1000.0 / prefill_ms);
        }
    } else {
        fprintf(stderr, "ds4: Mellum resident profile decode failed\n");
    }
    ds4_mellum_prefill_scratch_free(&depth_scratch);
    free(depth_toks);
    ds4_mellum_prefill_scratch_free(&prefill_scratch);
    free(prefill_toks);
    ds4_session_free(session);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_true_prefill_probe(ds4_engine *e, FILE *out) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    fprintf(stderr, "ds4: Mellum true-prefill probe requires Metal support\n");
    return 1;
#else
    static const int fixture_tokens[] = {
        27, 1397, 233, 12998, 497, 2717, 669, 60, 783, 846, 42, 99, 46,
        321, 800, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233,
    };
    const uint32_t n_tokens = (uint32_t)(sizeof(fixture_tokens) /
                                          sizeof(fixture_tokens[0]));
    if (!e || !out || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready ||
        !e->weights.output || e->weights.output->dim[1] == 0 ||
        e->weights.output->dim[1] > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "ds4: Mellum true-prefill probe requires an inspect-loaded Q8 Metal engine\n");
        return 1;
    }
    const uint64_t vocab_dim = e->weights.output->dim[1];
    const size_t logits_bytes = (size_t)vocab_dim * sizeof(float);
    float *decode_logits = xmalloc(logits_bytes);
    float *prefill_logits = xmalloc(logits_bytes);
    float *decode_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *prefill_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    const uint64_t layer_trace_bytes =
        (uint64_t)DS4_N_LAYER * DS4_N_EMBD * sizeof(float);
    float *decode_trace = xmalloc((size_t)layer_trace_bytes);
    float *prefill_trace = xmalloc((size_t)layer_trace_bytes);
    float *decode_attention_trace = xmalloc((size_t)layer_trace_bytes);
    float *prefill_attention_trace = xmalloc((size_t)layer_trace_bytes);
    ds4_gpu_tensor *decode_trace_gpu = ds4_gpu_tensor_alloc(layer_trace_bytes);
    ds4_gpu_tensor *prefill_trace_gpu = ds4_gpu_tensor_alloc(layer_trace_bytes);
    ds4_gpu_tensor *decode_attention_trace_gpu = ds4_gpu_tensor_alloc(layer_trace_bytes);
    ds4_gpu_tensor *prefill_attention_trace_gpu = ds4_gpu_tensor_alloc(layer_trace_bytes);
    ds4_mellum_decode_state *decode = ds4_mellum_decode_state_create(e, n_tokens);
    ds4_mellum_decode_state *prefill = ds4_mellum_decode_state_create(e, n_tokens);
    ds4_mellum_prefill_scratch scratch = {0};
    bool ok = decode_logits && prefill_logits && decode_hidden && prefill_hidden &&
              decode_trace && prefill_trace && decode_attention_trace &&
              prefill_attention_trace && decode_trace_gpu && prefill_trace_gpu &&
              decode_attention_trace_gpu && prefill_attention_trace_gpu &&
              decode && prefill &&
              ds4_mellum_decode_output_prepare(e, decode) &&
              ds4_mellum_decode_output_prepare(e, prefill) &&
              ds4_mellum_prefill_scratch_create(&scratch, n_tokens);
    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        ok = ds4_mellum_decode_token(e, decode, fixture_tokens[pos], NULL,
                                     NULL, NULL, NULL,
                                     pos + 1u == n_tokens ? decode_trace_gpu : NULL,
                                     pos + 1u == n_tokens ? decode_attention_trace_gpu : NULL,
                                     pos + 1u == n_tokens ? decode_logits : NULL,
                                     false, false);
    }
    if (ok && ds4_gpu_tensor_read(decode->hidden, 0, decode_hidden,
                                  (uint64_t)DS4_N_EMBD * sizeof(float)) == 0) {
        fprintf(stderr, "ds4: Mellum true-prefill probe could not read sequential hidden state\n");
        ok = false;
    }
    const uint32_t probe_chunk = ds4_mellum_probe_chunk(n_tokens, n_tokens);
    if (ok && !ds4_mellum_prefill_chunks(e, prefill, &scratch, fixture_tokens,
                                          n_tokens, probe_chunk,
                                          prefill_trace_gpu,
                                          prefill_attention_trace_gpu)) {
        fprintf(stderr, "ds4: Mellum true-prefill probe batch layer stack failed\n");
        ok = false;
    }
    if (ok && ds4_gpu_tensor_read(prefill->logits, 0, prefill_logits,
                                  logits_bytes) == 0) {
        fprintf(stderr, "ds4: Mellum true-prefill probe could not read batch logits\n");
        ok = false;
    }
    if (ok && (ds4_gpu_tensor_read(decode_trace_gpu, 0, decode_trace,
                                   layer_trace_bytes) == 0 ||
               ds4_gpu_tensor_read(prefill_trace_gpu, 0, prefill_trace,
                                   layer_trace_bytes) == 0 ||
               ds4_gpu_tensor_read(decode_attention_trace_gpu, 0,
                                   decode_attention_trace, layer_trace_bytes) == 0 ||
               ds4_gpu_tensor_read(prefill_attention_trace_gpu, 0,
                                   prefill_attention_trace, layer_trace_bytes) == 0)) {
        fprintf(stderr, "ds4: Mellum true-prefill probe could not read layer trace\n");
        ok = false;
    }
    ds4_gpu_tensor *prefill_last = NULL;
    if (ok) {
        /* The scratch rows hold only the final chunk, so the last token sits at
         * that chunk's last row rather than at n_tokens - 1. */
        const uint32_t last_chunk_tokens = n_tokens % probe_chunk ?
            n_tokens % probe_chunk : probe_chunk;
        ds4_gpu_tensor *last_rows = (DS4_N_LAYER & 1u) ?
            scratch.layer_out : scratch.hidden;
        prefill_last = ds4_gpu_tensor_view(
            last_rows,
            (uint64_t)(last_chunk_tokens - 1u) * DS4_N_EMBD * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float));
        if (!prefill_last) {
            fprintf(stderr, "ds4: Mellum true-prefill probe could not view batch hidden state\n");
            ok = false;
        } else if (ds4_gpu_tensor_read(prefill_last, 0, prefill_hidden,
                                        (uint64_t)DS4_N_EMBD * sizeof(float)) == 0) {
            fprintf(stderr, "ds4: Mellum true-prefill probe could not read batch hidden state\n");
            ok = false;
        }
    }
    ds4_gpu_tensor_free(prefill_last);
    float logit_max_abs = 0.0f, hidden_max_abs = 0.0f;
    double logit_sum_sq = 0.0, hidden_sum_sq = 0.0;
    uint32_t worst_layer = 0;
    float worst_layer_max_abs = 0.0f;
    double worst_layer_rms = 0.0;
    float worst_attention_max_abs = 0.0f;
    double worst_attention_rms = 0.0;
    if (ok) {
        for (uint64_t i = 0; i < vocab_dim; i++) {
            if (!isfinite(decode_logits[i]) || !isfinite(prefill_logits[i])) {
                ok = false;
                break;
            }
            const float delta = prefill_logits[i] - decode_logits[i];
            logit_max_abs = fmaxf(logit_max_abs, fabsf(delta));
            logit_sum_sq += (double)delta * delta;
        }
    }
    for (uint32_t i = 0; ok && i < DS4_N_EMBD; i++) {
        if (!isfinite(decode_hidden[i]) || !isfinite(prefill_hidden[i])) {
            ok = false;
            break;
        }
        const float delta = prefill_hidden[i] - decode_hidden[i];
        hidden_max_abs = fmaxf(hidden_max_abs, fabsf(delta));
        hidden_sum_sq += (double)delta * delta;
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        float layer_max_abs = 0.0f;
        float attention_max_abs = 0.0f;
        double layer_sum_sq = 0.0;
        double attention_sum_sq = 0.0;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            const float actual = prefill_trace[(uint64_t)il * DS4_N_EMBD + i];
            const float reference = decode_trace[(uint64_t)il * DS4_N_EMBD + i];
            const float attention_actual = prefill_attention_trace[
                (uint64_t)il * DS4_N_EMBD + i];
            const float attention_reference = decode_attention_trace[
                (uint64_t)il * DS4_N_EMBD + i];
            if (!isfinite(actual) || !isfinite(reference) ||
                !isfinite(attention_actual) || !isfinite(attention_reference)) {
                ok = false;
                break;
            }
            const float delta = actual - reference;
            const float attention_delta = attention_actual - attention_reference;
            layer_max_abs = fmaxf(layer_max_abs, fabsf(delta));
            layer_sum_sq += (double)delta * delta;
            attention_max_abs = fmaxf(attention_max_abs, fabsf(attention_delta));
            attention_sum_sq += (double)attention_delta * attention_delta;
        }
        const double layer_rms = sqrt(layer_sum_sq / (double)DS4_N_EMBD);
        const double attention_rms = sqrt(attention_sum_sq / (double)DS4_N_EMBD);
        if (layer_rms > worst_layer_rms) {
            worst_layer = il;
            worst_layer_max_abs = layer_max_abs;
            worst_layer_rms = layer_rms;
        }
        if (attention_rms > worst_attention_rms) {
            worst_attention_max_abs = attention_max_abs;
            worst_attention_rms = attention_rms;
        }
        if (ok) fprintf(out,
                        "Mellum true-prefill layer=%u attention_max=%g attention_rms=%g layer_max=%g layer_rms=%g\n",
                        il, attention_max_abs, attention_rms,
                        layer_max_abs, layer_rms);
    }
    const double logit_rms = ok ? sqrt(logit_sum_sq / (double)vocab_dim) : 0.0;
    const double hidden_rms = ok ? sqrt(hidden_sum_sq / (double)DS4_N_EMBD) : 0.0;
    if (!ok) {
        fprintf(stderr, "ds4: Mellum true-prefill probe execution failed\n");
    } else {
        fprintf(out,
                "Mellum true-prefill probe tokens=%u chunk=%u layers=%u hidden max_abs=%g rms=%g logits max_abs=%g rms=%g\n",
                n_tokens, probe_chunk, DS4_N_LAYER, hidden_max_abs, hidden_rms,
                logit_max_abs, logit_rms);
        fprintf(out, "Mellum true-prefill worst-layer=%u max_abs=%g rms=%g\n",
                worst_layer, worst_layer_max_abs, worst_layer_rms);
        fprintf(out, "Mellum true-prefill worst-attention max_abs=%g rms=%g\n",
                worst_attention_max_abs, worst_attention_rms);
    }
    ds4_mellum_prefill_scratch_free(&scratch);
    ds4_gpu_tensor_free(prefill_attention_trace_gpu);
    ds4_gpu_tensor_free(decode_attention_trace_gpu);
    ds4_gpu_tensor_free(prefill_trace_gpu);
    ds4_gpu_tensor_free(decode_trace_gpu);
    ds4_mellum_decode_state_free(prefill);
    ds4_mellum_decode_state_free(decode);
    free(prefill_trace);
    free(decode_trace);
    free(prefill_attention_trace);
    free(decode_attention_trace);
    free(prefill_hidden);
    free(decode_hidden);
    free(prefill_logits);
    free(decode_logits);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_mellum_true_prefill_swa_probe(ds4_engine *e, FILE *out) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)out;
    fprintf(stderr, "ds4: Mellum true-prefill SWA probe requires Metal support\n");
    return 1;
#else
    const uint32_t tail_tokens = 6u;
    if (DS4_N_SWA == 0 || DS4_N_SWA > UINT32_MAX - tail_tokens) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe has an invalid sliding window\n");
        return 1;
    }
    const uint32_t first_chunk = DS4_N_SWA;
    const uint32_t n_tokens = first_chunk + tail_tokens;
    const uint32_t small_chunk = ds4_mellum_probe_chunk(32u, n_tokens);
    if (!e || !out || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_MELLUM ||
        e->backend != DS4_BACKEND_METAL || !e->mellum_decode_contract_ready ||
        !e->weights.output || e->weights.output->dim[1] == 0 ||
        e->weights.output->dim[1] > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe requires an inspect-loaded Q8 Metal engine\n");
        return 1;
    }
    int *tokens = xmalloc((size_t)n_tokens * sizeof(*tokens));
    for (uint32_t i = 0; i < n_tokens; i++) {
        tokens[i] = (int)(((uint64_t)i * 7919u + 27u) % DS4_N_VOCAB);
    }
    const uint64_t vocab_dim = e->weights.output->dim[1];
    const size_t logits_bytes = (size_t)vocab_dim * sizeof(float);
    const uint64_t hidden_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    float *decode_logits = xmalloc(logits_bytes);
    float *prefill_logits = xmalloc(logits_bytes);
    float *decode_window_logits = xmalloc(logits_bytes);
    float *prefill_window_logits = xmalloc(logits_bytes);
    float *decode_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *prefill_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *decode_window_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *prefill_window_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *small_logits = xmalloc(logits_bytes);
    float *small_hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    ds4_mellum_decode_state *decode = ds4_mellum_decode_state_create(e, n_tokens);
    ds4_mellum_decode_state *prefill = ds4_mellum_decode_state_create(e, n_tokens);
    ds4_mellum_decode_state *small = ds4_mellum_decode_state_create(e, n_tokens);
    ds4_mellum_prefill_scratch scratch = {0};
    ds4_mellum_prefill_scratch small_scratch = {0};
    bool ok = tokens && decode_logits && prefill_logits && decode_window_logits &&
              prefill_window_logits && decode_hidden && prefill_hidden &&
              decode_window_hidden && prefill_window_hidden && small_logits &&
              small_hidden && decode && prefill && small &&
              ds4_mellum_decode_output_prepare(e, decode) &&
              ds4_mellum_decode_output_prepare(e, prefill) &&
              ds4_mellum_decode_output_prepare(e, small) &&
              ds4_mellum_prefill_scratch_create(&scratch, first_chunk) &&
              ds4_mellum_prefill_scratch_create(&small_scratch, small_chunk);
    for (uint32_t pos = 0; ok && pos < n_tokens; pos++) {
        const bool window_last = pos + 1u == first_chunk;
        const bool final_last = pos + 1u == n_tokens;
        ok = ds4_mellum_decode_token(e, decode, tokens[pos], NULL, NULL,
                                     NULL, NULL, NULL, NULL,
                                     window_last ? decode_window_logits :
                                     (final_last ? decode_logits : NULL),
                                     false, false);
        if (ok && window_last && ds4_gpu_tensor_read(
                decode->hidden, 0, decode_window_hidden, hidden_bytes) == 0) {
            fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read sequential window hidden state\n");
            ok = false;
        }
    }
    if (ok && ds4_gpu_tensor_read(decode->hidden, 0, decode_hidden,
                                  hidden_bytes) == 0) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read sequential hidden state\n");
        ok = false;
    }
    if (ok && !ds4_mellum_prefill_tokens(e, prefill, &scratch, tokens,
                                          first_chunk, NULL, NULL, true)) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe first chunk failed\n");
        ok = false;
    }
    if (ok && ds4_gpu_tensor_read(prefill->logits, 0, prefill_window_logits,
                                  logits_bytes) == 0) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read batch window logits\n");
        ok = false;
    }
    ds4_gpu_tensor *prefill_window_last = NULL;
    if (ok) {
        ds4_gpu_tensor *last_rows = (DS4_N_LAYER & 1u) ?
            scratch.layer_out : scratch.hidden;
        prefill_window_last = ds4_gpu_tensor_view(
            last_rows, (uint64_t)(first_chunk - 1u) * hidden_bytes, hidden_bytes);
        if (!prefill_window_last || ds4_gpu_tensor_read(
                prefill_window_last, 0, prefill_window_hidden, hidden_bytes) == 0) {
            fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read batch window hidden state\n");
            ok = false;
        }
    }
    ds4_gpu_tensor_free(prefill_window_last);
    if (ok && !ds4_mellum_prefill_tokens(e, prefill, &scratch,
                                          tokens + first_chunk, tail_tokens,
                                          NULL, NULL, true)) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe tail chunk failed\n");
        ok = false;
    }
    if (ok && ds4_gpu_tensor_read(prefill->logits, 0, prefill_logits,
                                  logits_bytes) == 0) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read batch logits\n");
        ok = false;
    }
    ds4_gpu_tensor *prefill_last = NULL;
    if (ok) {
        ds4_gpu_tensor *last_rows = (DS4_N_LAYER & 1u) ?
            scratch.layer_out : scratch.hidden;
        prefill_last = ds4_gpu_tensor_view(
            last_rows, (uint64_t)(tail_tokens - 1u) * DS4_N_EMBD * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float));
        if (!prefill_last || ds4_gpu_tensor_read(
                prefill_last, 0, prefill_hidden,
                (uint64_t)DS4_N_EMBD * sizeof(float)) == 0) {
            fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read batch hidden state\n");
            ok = false;
        }
    }
    ds4_gpu_tensor_free(prefill_last);
    if (ok && !ds4_mellum_prefill_chunks(e, small, &small_scratch, tokens,
                                          n_tokens, small_chunk, NULL, NULL)) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe chunked schedule failed\n");
        ok = false;
    }
    if (ok && ds4_gpu_tensor_read(small->logits, 0, small_logits,
                                  logits_bytes) == 0) {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read chunked logits\n");
        ok = false;
    }
    ds4_gpu_tensor *small_last = NULL;
    if (ok) {
        const uint32_t small_last_tokens = n_tokens % small_chunk ?
            n_tokens % small_chunk : small_chunk;
        ds4_gpu_tensor *last_rows = (DS4_N_LAYER & 1u) ?
            small_scratch.layer_out : small_scratch.hidden;
        small_last = ds4_gpu_tensor_view(
            last_rows, (uint64_t)(small_last_tokens - 1u) * hidden_bytes,
            hidden_bytes);
        if (!small_last || ds4_gpu_tensor_read(
                small_last, 0, small_hidden, hidden_bytes) == 0) {
            fprintf(stderr, "ds4: Mellum true-prefill SWA probe could not read chunked hidden state\n");
            ok = false;
        }
    }
    ds4_gpu_tensor_free(small_last);
    float window_hidden_max_abs = 0.0f, window_logit_max_abs = 0.0f;
    double window_hidden_sum_sq = 0.0, window_logit_sum_sq = 0.0;
    for (uint32_t i = 0; ok && i < DS4_N_EMBD; i++) {
        if (!isfinite(decode_window_hidden[i]) ||
            !isfinite(prefill_window_hidden[i])) {
            ok = false;
            break;
        }
        const float delta = prefill_window_hidden[i] - decode_window_hidden[i];
        window_hidden_max_abs = fmaxf(window_hidden_max_abs, fabsf(delta));
        window_hidden_sum_sq += (double)delta * delta;
    }
    for (uint64_t i = 0; ok && i < vocab_dim; i++) {
        if (!isfinite(decode_window_logits[i]) ||
            !isfinite(prefill_window_logits[i])) {
            ok = false;
            break;
        }
        const float delta = prefill_window_logits[i] - decode_window_logits[i];
        window_logit_max_abs = fmaxf(window_logit_max_abs, fabsf(delta));
        window_logit_sum_sq += (double)delta * delta;
    }
    float hidden_max_abs = 0.0f, logit_max_abs = 0.0f;
    double hidden_sum_sq = 0.0, logit_sum_sq = 0.0;
    float small_hidden_max_abs = 0.0f, small_logit_max_abs = 0.0f;
    double small_hidden_sum_sq = 0.0, small_logit_sum_sq = 0.0;
    for (uint32_t i = 0; ok && i < DS4_N_EMBD; i++) {
        if (!isfinite(decode_hidden[i]) || !isfinite(prefill_hidden[i])) {
            ok = false;
            break;
        }
        const float delta = prefill_hidden[i] - decode_hidden[i];
        hidden_max_abs = fmaxf(hidden_max_abs, fabsf(delta));
        hidden_sum_sq += (double)delta * delta;
    }
    for (uint64_t i = 0; ok && i < vocab_dim; i++) {
        if (!isfinite(decode_logits[i]) || !isfinite(prefill_logits[i])) {
            ok = false;
            break;
        }
        const float delta = prefill_logits[i] - decode_logits[i];
        logit_max_abs = fmaxf(logit_max_abs, fabsf(delta));
        logit_sum_sq += (double)delta * delta;
    }
    for (uint32_t i = 0; ok && i < DS4_N_EMBD; i++) {
        if (!isfinite(decode_hidden[i]) || !isfinite(small_hidden[i])) {
            ok = false;
            break;
        }
        const float delta = small_hidden[i] - decode_hidden[i];
        small_hidden_max_abs = fmaxf(small_hidden_max_abs, fabsf(delta));
        small_hidden_sum_sq += (double)delta * delta;
    }
    for (uint64_t i = 0; ok && i < vocab_dim; i++) {
        if (!isfinite(decode_logits[i]) || !isfinite(small_logits[i])) {
            ok = false;
            break;
        }
        const float delta = small_logits[i] - decode_logits[i];
        small_logit_max_abs = fmaxf(small_logit_max_abs, fabsf(delta));
        small_logit_sum_sq += (double)delta * delta;
    }
    if (ok) {
        fprintf(out,
                "Mellum true-prefill SWA probe window=%u hidden max_abs=%g rms=%g logits max_abs=%g rms=%g\n",
                first_chunk, window_hidden_max_abs,
                sqrt(window_hidden_sum_sq / (double)DS4_N_EMBD),
                window_logit_max_abs,
                sqrt(window_logit_sum_sq / (double)vocab_dim));
        fprintf(out,
                "Mellum true-prefill SWA probe tokens=%u chunks=%u+%u hidden max_abs=%g rms=%g logits max_abs=%g rms=%g\n",
                n_tokens, first_chunk, tail_tokens, hidden_max_abs,
                sqrt(hidden_sum_sq / (double)DS4_N_EMBD), logit_max_abs,
                sqrt(logit_sum_sq / (double)vocab_dim));
        fprintf(out,
                "Mellum true-prefill SWA probe tokens=%u chunks=%u hidden max_abs=%g rms=%g logits max_abs=%g rms=%g\n",
                n_tokens, small_chunk, small_hidden_max_abs,
                sqrt(small_hidden_sum_sq / (double)DS4_N_EMBD),
                small_logit_max_abs,
                sqrt(small_logit_sum_sq / (double)vocab_dim));
    } else {
        fprintf(stderr, "ds4: Mellum true-prefill SWA probe execution failed\n");
    }
    ds4_mellum_prefill_scratch_free(&scratch);
    ds4_mellum_prefill_scratch_free(&small_scratch);
    ds4_mellum_decode_state_free(small);
    ds4_mellum_decode_state_free(prefill);
    ds4_mellum_decode_state_free(decode);
    free(prefill_window_hidden);
    free(decode_window_hidden);
    free(small_hidden);
    free(prefill_hidden);
    free(decode_hidden);
    free(prefill_window_logits);
    free(decode_window_logits);
    free(small_logits);
    free(prefill_logits);
    free(decode_logits);
    free(tokens);
    return ok ? 0 : 1;
#endif
}
