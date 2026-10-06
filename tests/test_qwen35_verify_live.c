/* G3 of the DFlash spec (forced drafts), on the dense qwen35 model.
 *
 * For each prompt and depth (about 100, 2,600 and 9,700 tokens) a session in
 * DFlash mode syncs a prefix, saves a snapshot and takes the greedy
 * continuation x0..x7 with each step's runner-up.  Case k = 0..7 drafts
 * x1..xk, then the runner-up of step k + 1 (none at k = 7), then filler up to
 * 7 drafts.  Reference: from the snapshot, x0 and the 7 drafts evaluated one
 * at a time, keeping all 8 logit rows and the payload after k + 1 tokens.
 * Candidate: from the snapshot, one verify of the same 8 tokens, each row
 * selected and compared bit for bit, the open verify's refusals checked, the
 * first k + 1 rows committed and the payload compared byte for byte.  One
 * JSON line per case; exit status 0 only when every case passes.
 *
 * Usage: test_qwen35_verify_live MODEL PROMPTS FILLER [--first I] [--prompts N]
 *                                [--depths D,D,...] [--ctx N]
 * PROMPTS holds JSON lines with "name" and "prompt"; a prefix is FILLER's
 * tokens, repeated as needed, then the prompt's chat with thinking on, so
 * a prefix is the chat alone when the chat is longer than the depth. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROWS 8

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static void die(const char *what, const char *err) {
    fprintf(stderr, "test_qwen35_verify_live: %s%s%s\n", what, err && err[0] ? ": " : "", err ? err : "");
    exit(2);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("cannot open", path);
    if (fseek(fp, 0, SEEK_END) != 0) die("cannot seek", path);
    const long n = ftell(fp);
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) die("cannot size", path);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) die("cannot read", path);
    buf[n] = 0;
    fclose(fp);
    return buf;
}

static void put_utf8(char **o, uint32_t c) {
    char *p = *o;
    if (c < 0x80) *p++ = (char)c;
    else if (c < 0x800) { *p++ = (char)(0xc0 | c >> 6); *p++ = (char)(0x80 | (c & 0x3f)); }
    else if (c < 0x10000) {
        *p++ = (char)(0xe0 | c >> 12); *p++ = (char)(0x80 | ((c >> 6) & 0x3f)); *p++ = (char)(0x80 | (c & 0x3f));
    } else {
        *p++ = (char)(0xf0 | c >> 18); *p++ = (char)(0x80 | ((c >> 12) & 0x3f));
        *p++ = (char)(0x80 | ((c >> 6) & 0x3f)); *p++ = (char)(0x80 | (c & 0x3f));
    }
    *o = p;
}

static uint32_t hex4(const char *s) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = s[i];
        v = v * 16u + (uint32_t)(c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                                 c >= 'A' && c <= 'F' ? c - 'A' + 10 : 0);
    }
    return v;
}

/* The string value of "key" in one JSON object line (escapes decoded), or
 * NULL.  Enough for the prompts file: flat objects of string fields. */
static char *json_field(const char *line, const char *key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(line, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ' || *p == ':') p++;
    if (*p++ != '"') return NULL;
    char *out = malloc(strlen(p) * 2 + 4), *o = out;
    while (*p && *p != '"') {
        if (*p != '\\') { *o++ = *p++; continue; }
        p++;
        switch (*p) {
        case 'n': *o++ = '\n'; p++; break;
        case 't': *o++ = '\t'; p++; break;
        case 'r': *o++ = '\r'; p++; break;
        case 'b': *o++ = '\b'; p++; break;
        case 'f': *o++ = '\f'; p++; break;
        case 'u': {
            uint32_t c = hex4(p + 1);
            p += 5;
            if (c >= 0xd800 && c < 0xdc00 && p[0] == '\\' && p[1] == 'u') {
                c = 0x10000 + ((c - 0xd800) << 10) + (hex4(p + 2) - 0xdc00);
                p += 6;
            }
            put_utf8(&o, c);
            break;
        }
        default: *o++ = *p++; break;
        }
    }
    *o = 0;
    return out;
}

static uint64_t fnv1a(const uint8_t *p, uint64_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

static void sync_or_die(ds4_session *s, const ds4_tokens *t) {
    char err[256] = {0};
    if (ds4_session_sync(s, t, err, sizeof(err)) != 0) die("sync failed", err);
}

static void load_or_die(ds4_session *s, const ds4_session_snapshot *snap) {
    char err[256] = {0};
    if (ds4_session_load_snapshot(s, snap, err, sizeof(err)) != 0) die("snapshot load failed", err);
}

static void save_or_die(ds4_session *s, ds4_session_snapshot *snap) {
    char err[256] = {0};
    if (ds4_session_save_snapshot(s, snap, err, sizeof(err)) != 0) die("snapshot save failed", err);
}

static void eval_or_die(ds4_session *s, int token) {
    char err[256] = {0};
    if (ds4_session_eval(s, token, err, sizeof(err)) != 0) die("eval failed", err);
}

/* While a verify is open, each call that would move or read out the session
 * must refuse and name the verify, and rewind must leave the position. */
static bool open_verify_refuses(ds4_session *s, const ds4_tokens *prefix, ds4_session_snapshot *scratch,
                                const ds4_session_snapshot *base, int token) {
    char err[256];
    bool ok = true;
    const int pos = ds4_session_pos(s);
    err[0] = 0;
    ok = ok && ds4_session_eval(s, token, err, sizeof(err)) != 0 && strstr(err, "verify");
    err[0] = 0;
    ok = ok && ds4_session_sync(s, prefix, err, sizeof(err)) != 0 && strstr(err, "verify");
    err[0] = 0;
    ok = ok && ds4_session_save_snapshot(s, scratch, err, sizeof(err)) != 0 && strstr(err, "verify");
    err[0] = 0;
    ok = ok && ds4_session_load_snapshot(s, base, err, sizeof(err)) != 0 && strstr(err, "verify");
    err[0] = 0;
    ok = ok && ds4_session_verify(s, token, NULL, 0, err, sizeof(err)) != 0 && strstr(err, "verify");
    ok = ok && ds4_session_set_verify_mode(s, DS4_VERIFY_MODE_OFF) != 0;
    ds4_session_rewind(s, pos > 0 ? pos - 1 : 0);
    ok = ok && ds4_session_pos(s) == pos;
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s MODEL PROMPTS FILLER [--first I] [--prompts N] [--depths D,D,...] [--ctx N]\n",
                argv[0]);
        return 2;
    }
    int first = 0, count = 1 << 30, ctx = 10240, depths[8] = {100, 2600, 9700}, n_depths = 3;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--first")) first = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--prompts")) count = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--depths")) {
            n_depths = 0;
            for (char *p = argv[i + 1]; *p && n_depths < 8;) {
                depths[n_depths++] = (int)strtol(p, &p, 10);
                if (*p == ',') p++;
            }
        } else die("unknown option", argv[i]);
    }

    ds4_engine_options opt = {.model_path = argv[1], .context_size = ctx, .backend = DS4_BACKEND_METAL};
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("engine open failed", argv[1]);
    if (ds4_engine_verify_rows(engine) != ROWS) die("the model has no 8-row verify", argv[1]);
    const int vocab = ds4_engine_vocab_size(engine);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, ctx) != 0) die("session create failed", NULL);
    if (ds4_session_set_verify_mode(s, DS4_VERIFY_MODE_MMA) != 0) die("DFlash mode refused", NULL);

    char *filler_text = read_file(argv[3]);
    ds4_tokens filler = {0};
    ds4_tokenize_text(engine, filler_text, &filler);
    free(filler_text);
    if (filler.len < ROWS) die("filler too short", argv[3]);

    float *ref = malloc((size_t)ROWS * vocab * sizeof(float)), *row = malloc((size_t)vocab * sizeof(float));
    ds4_session_snapshot base = {0}, snap_ref = {0}, snap_cand = {0}, scratch = {0};
    char *prompts = read_file(argv[2]);
    int index = 0, cases = 0, passed = 0;
    for (char *line = strtok(prompts, "\n"); line; line = strtok(NULL, "\n"), index++) {
        if (index < first || index >= first + count) continue;
        char *name = json_field(line, "name"), *prompt = json_field(line, "prompt");
        if (!name || !prompt) die("prompt line without name or prompt", line);
        ds4_tokens chat = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_HIGH, &chat);
        for (int d = 0; d < n_depths; d++) {
            ds4_tokens prefix = {0};
            for (int i = 0; i + chat.len < depths[d]; i++) ds4_tokens_push(&prefix, filler.v[i % filler.len]);
            for (int i = 0; i < chat.len; i++) ds4_tokens_push(&prefix, chat.v[i]);
            if (prefix.len + ROWS + 1 > ctx) die("prefix does not fit the context", name);
            sync_or_die(s, &prefix);
            save_or_die(s, &base);

            int x[ROWS], runner[ROWS];
            x[0] = ds4_session_argmax(s);
            runner[0] = -1;
            for (int i = 1; i < ROWS; i++) {
                eval_or_die(s, x[i - 1]);
                x[i] = ds4_session_argmax(s);
                runner[i] = ds4_session_argmax_excluding(s, x[i]);
            }

            for (int k = 0; k < ROWS; k++) {
                int tokens[ROWS];
                tokens[0] = x[0];
                for (int j = 1; j < ROWS; j++) {
                    tokens[j] = j <= k ? x[j] : j == k + 1 ? runner[j] : filler.v[(prefix.len + j) % filler.len];
                }

                load_or_die(s, &base);
                double serial_ms = 0.0;
                for (int t = 0; t < ROWS; t++) {
                    const double t0 = now_ms();
                    eval_or_die(s, tokens[t]);
                    serial_ms += now_ms() - t0;
                    if (ds4_session_copy_logits(s, ref + (size_t)t * vocab, vocab) != vocab) die("copy logits", NULL);
                    if (t == k) save_or_die(s, &snap_ref);
                }

                load_or_die(s, &base);
                char err[256] = {0};
                double t0 = now_ms();
                if (ds4_session_verify(s, tokens[0], tokens + 1, ROWS - 1, err, sizeof(err)) != 0)
                    die("verify failed", err);
                const double verify_ms = now_ms() - t0;
                int rows_bitwise = 0, first_diff = -1;
                float max_abs = 0.0f;
                for (int r = 0; r < ROWS; r++) {
                    if (ds4_session_verify_select(s, r) != 0) die("select refused", NULL);
                    if (ds4_session_copy_logits(s, row, vocab) != vocab) die("copy logits", NULL);
                    const float *want = ref + (size_t)r * vocab;
                    if (!memcmp(row, want, (size_t)vocab * sizeof(float))) rows_bitwise++;
                    else if (first_diff < 0) first_diff = r;
                    for (int i = 0; i < vocab; i++) max_abs = fmaxf(max_abs, fabsf(row[i] - want[i]));
                }
                const bool refusals = open_verify_refuses(s, &prefix, &scratch, &base, tokens[0]);
                t0 = now_ms();
                if (ds4_session_verify_commit(s, k + 1, err, sizeof(err)) != 0) die("commit failed", err);
                const double commit_ms = now_ms() - t0;
                save_or_die(s, &snap_cand);
                const bool payload_equal = snap_ref.len == snap_cand.len &&
                                           !memcmp(snap_ref.ptr, snap_cand.ptr, snap_ref.len);
                const bool pass = rows_bitwise == ROWS && payload_equal && refusals &&
                                  ds4_session_pos(s) == prefix.len + k + 1;
                cases++;
                passed += pass;
                printf("{\"prompt\":\"%s\",\"depth\":%d,\"pos0\":%d,\"k\":%d,\"rows_bitwise\":%d,"
                       "\"first_diff_row\":%d,\"max_abs_diff\":%g,\"payload_equal\":%s,\"payload_bytes\":%llu,"
                       "\"payload_fnv_ref\":\"%016llx\",\"payload_fnv_cand\":\"%016llx\",\"refusals\":%s,"
                       "\"serial_ms\":%.2f,\"verify_ms\":%.2f,\"commit_ms\":%.2f,\"pass\":%s}\n",
                       name, depths[d], prefix.len, k, rows_bitwise, first_diff, (double)max_abs,
                       payload_equal ? "true" : "false", (unsigned long long)snap_cand.len,
                       (unsigned long long)fnv1a(snap_ref.ptr, snap_ref.len),
                       (unsigned long long)fnv1a(snap_cand.ptr, snap_cand.len),
                       refusals ? "true" : "false", serial_ms, verify_ms, commit_ms, pass ? "true" : "false");
                fflush(stdout);
            }
            ds4_tokens_free(&prefix);
        }
        ds4_tokens_free(&chat);
        free(prompt);
        free(name);
    }
    ds4_spec_stats st;
    ds4_session_spec_stats(s, &st);
    fprintf(stderr, "test_qwen35_verify_live: steps %llu drafted %llu accepted %llu verify_ms %.1f commit_ms %.1f\n",
            (unsigned long long)st.steps, (unsigned long long)st.drafted, (unsigned long long)st.accepted,
            st.verify_ms, st.commit_ms);
    fprintf(stderr, "G3: %d/%d cases pass\n", passed, cases);
    ds4_session_snapshot_free(&scratch);
    ds4_session_snapshot_free(&snap_cand);
    ds4_session_snapshot_free(&snap_ref);
    ds4_session_snapshot_free(&base);
    free(prompts);
    free(row);
    free(ref);
    ds4_tokens_free(&filler);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return cases > 0 && passed == cases ? 0 : 1;
}
