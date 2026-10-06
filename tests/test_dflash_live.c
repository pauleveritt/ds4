/* Live checks of the DFlash drafter's state on the dense qwen35 model (the
 * DFlash plan's S4.1, S4.2 and G7).
 *
 * capture: per prompt and depth, a session in DFlash mode syncs a prefix of
 * FILLER's tokens and the prompt's chat, saves a snapshot and evaluates the
 * greedy continuation x0..x7 one token at a time, keeping each eval row's
 * captured features; from the snapshot one verify of x0..x7 captures its 8
 * rows, which must equal the eval rows bit for bit, and the commit of all 8
 * must leave them.  The oldest row the sync captures (2,047 positions
 * before the prefix's end, or row 0) must be held, and a snapshot load must
 * forget the rows.  One JSON line per case; exit status 0 only when every
 * case passes.
 *
 * features: against llama.cpp's dump (llama.cpp bench/dflash-dump, index.jsonl
 * and features.f32), per prompt our session syncs the prompt's ids and
 * evaluates the dump's generated tokens one at a time in DFlash mode; every
 * position's features are compared with llama.cpp's, tap by tap, by cosine.
 * Exit status 0 when every tap of every generated position is >= 0.999.
 *
 * inject: see run_inject; the recompute is the host's s4 data dir's
 * inject_check.py.  g7 and spec: see run_g7 and run_spec.  payload-save and
 * payload-load: see run_payload_save; the comparison is the host's s6 data
 * dir's payload_check.py.
 *
 * Usage: test_dflash_live capture MODEL DRAFTER PROMPTS FILLER [--prompts N]
 *                                 [--depths D,D,...] [--ctx N]
 *        test_dflash_live features MODEL DRAFTER INDEX FEATURES [--prompts N]
 *                                 [--first I] [--ctx N]
 *        test_dflash_live inject MODEL DRAFTER PROMPTS FILLER OUT [--first I]
 *                                 [--depths D] [--ctx N]
 *        test_dflash_live g7 MODEL DRAFTER INDEX FEATURES LATTICE [--prompts N]
 *                                 [--first I] [--ctx N]
 *        test_dflash_live spec MODEL DRAFTER PROMPTS FILLER [--prompts N] [--first I]
 *                                 [--depths D] [--ctx N] [--tokens M]
 *        test_dflash_live payload-save MODEL DRAFTER PROMPTS FILLER OUT [--first I]
 *                                 [--depths D] [--ctx N] [--tokens M]
 *        test_dflash_live payload-load MODEL DRAFTER|- IN OUT|- [--ctx N] [--tokens M]
 * PROMPTS holds JSON lines with "name" and "prompt", as for
 * test_qwen35_verify_live. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <stdbool.h>
#include <sys/types.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROWS 8

static void die(const char *what, const char *err) {
    fprintf(stderr, "test_dflash_live: %s%s%s\n", what, err && err[0] ? ": " : "", err ? err : "");
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

/* The string value of "key" in one flat JSON object line, escapes decoded,
 * or NULL. */
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

static void sync_or_die(ds4_session *s, const ds4_tokens *t) {
    char err[256] = {0};
    if (ds4_session_sync(s, t, err, sizeof(err)) != 0) die("sync failed", err);
}

static void eval_or_die(ds4_session *s, int token) {
    char err[256] = {0};
    if (ds4_session_eval(s, token, err, sizeof(err)) != 0) die("eval failed", err);
}

static void features_or_die(ds4_session *s, int pos, float *out) {
    if (ds4_session_dflash_features(s, pos, out) != 0) die("no captured features at a position", NULL);
}

typedef struct {
    const char *model, *drafter;
    int ctx, first, count, n_depths, depths[8], tokens;
    char **rest;
    int n_rest;
} live_args;

static live_args parse_args(int argc, char **argv, int n_fixed) {
    live_args a = { .model = argv[2], .drafter = argv[3], .ctx = 10240, .count = 1 << 30,
                    .n_depths = 3, .depths = {100, 2600, 9700}, .tokens = 128 };
    a.rest = argv + 4;
    a.n_rest = n_fixed;
    for (int i = 4 + n_fixed; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--prompts")) a.count = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--first")) a.first = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--ctx")) a.ctx = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--tokens")) a.tokens = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--depths")) {
            a.n_depths = 0;
            for (char *p = argv[i + 1]; *p && a.n_depths < 8;) {
                a.depths[a.n_depths++] = (int)strtol(p, &p, 10);
                if (*p == ',') p++;
            }
        } else die("unknown option", argv[i]);
    }
    return a;
}

/* the engine with the drafter, or without one when DRAFTER is "-" */
static ds4_engine *open_or_die(const live_args *a) {
    const bool none = !strcmp(a->drafter, "-");
    ds4_engine_options opt = {.model_path = a->model, .mtp_path = none ? NULL : a->drafter,
                              .context_size = a->ctx, .backend = DS4_BACKEND_METAL};
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("engine open failed", a->model);
    if (ds4_engine_has_dflash(engine) == none) die(none ? "the engine has a drafter" : "the engine has no drafter",
                                                   a->drafter);
    return engine;
}

static int run_capture(int argc, char **argv) {
    if (argc < 6) die("capture needs MODEL DRAFTER PROMPTS FILLER", NULL);
    const live_args a = parse_args(argc, argv, 2);
    ds4_engine *engine = open_or_die(&a);
    const int width = ds4_engine_dflash_feature_floats(engine);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0) die("session create failed", NULL);
    if (ds4_session_set_verify_mode(s, DS4_VERIFY_MODE_MMA) != 0) die("DFlash mode refused", NULL);

    char *filler_text = read_file(a.rest[1]);
    ds4_tokens filler = {0};
    ds4_tokenize_text(engine, filler_text, &filler);
    free(filler_text);
    if (filler.len < ROWS) die("filler too short", a.rest[1]);

    float *evalf = malloc((size_t)ROWS * width * sizeof(float)), *row = malloc((size_t)width * sizeof(float));
    ds4_session_snapshot base = {0};
    char *prompts = read_file(a.rest[0]);
    int index = 0, cases = 0, passed = 0;
    for (char *line = strtok(prompts, "\n"); line; line = strtok(NULL, "\n"), index++) {
        if (index < a.first || index >= a.first + a.count) continue;
        char *name = json_field(line, "name"), *prompt = json_field(line, "prompt");
        if (!name || !prompt) die("prompt line without name or prompt", line);
        ds4_tokens chat = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_HIGH, &chat);
        for (int d = 0; d < a.n_depths; d++) {
            ds4_tokens prefix = {0};
            for (int i = 0; i + chat.len < a.depths[d]; i++) ds4_tokens_push(&prefix, filler.v[i % filler.len]);
            for (int i = 0; i < chat.len; i++) ds4_tokens_push(&prefix, chat.v[i]);
            if (prefix.len + ROWS + 1 > a.ctx) die("prefix does not fit the context", name);
            const int n = prefix.len;
            sync_or_die(s, &prefix);
            const int oldest = n >= 2048 ? n - 2047 : 0;
            bool window_ok = ds4_session_dflash_features(s, oldest, row) == 0;
            int nonzero = 0;
            for (int i = 0; window_ok && i < width; i++) nonzero += row[i] != 0.0f;
            window_ok = window_ok && nonzero > width / 2;
            const char *window = window_ok ? "held" : "missing";
            char err[256] = {0};
            if (ds4_session_save_snapshot(s, &base, err, sizeof(err)) != 0) die("snapshot save failed", err);

            int x[ROWS];
            x[0] = ds4_session_argmax(s);
            for (int i = 0; i < ROWS; i++) {
                eval_or_die(s, x[i]);
                features_or_die(s, n + i, evalf + (size_t)i * width);
                if (i + 1 < ROWS) x[i + 1] = ds4_session_argmax(s);
            }

            if (ds4_session_load_snapshot(s, &base, err, sizeof(err)) != 0) die("snapshot load failed", err);
            const bool forgotten = ds4_session_dflash_features(s, n, row) != 0;
            if (ds4_session_verify(s, x[0], x + 1, ROWS - 1, err, sizeof(err)) != 0) die("verify failed", err);
            int verify_bitwise = 0, first_diff = -1;
            double max_abs = 0.0;
            for (int r = 0; r < ROWS; r++) {
                features_or_die(s, n + r, row);
                const float *want = evalf + (size_t)r * width;
                if (!memcmp(row, want, (size_t)width * sizeof(float))) verify_bitwise++;
                else if (first_diff < 0) first_diff = r;
                for (int i = 0; i < width; i++) max_abs = fmax(max_abs, fabs((double)row[i] - want[i]));
            }
            if (ds4_session_verify_commit(s, ROWS, err, sizeof(err)) != 0) die("commit failed", err);
            int commit_bitwise = 0;
            for (int r = 0; r < ROWS; r++) {
                features_or_die(s, n + r, row);
                commit_bitwise += !memcmp(row, evalf + (size_t)r * width, (size_t)width * sizeof(float));
            }
            const bool pass = verify_bitwise == ROWS && commit_bitwise == ROWS && window_ok && forgotten;
            cases++;
            passed += pass;
            printf("{\"prompt\":\"%s\",\"depth\":%d,\"pos0\":%d,\"verify_rows_bitwise\":%d,\"first_diff_row\":%d,"
                   "\"max_abs_diff\":%g,\"after_commit_bitwise\":%d,\"sync_window\":\"%s\","
                   "\"load_forgets\":%s,\"pass\":%s}\n",
                   name, a.depths[d], n, verify_bitwise, first_diff, max_abs, commit_bitwise, window,
                   forgotten ? "true" : "false", pass ? "true" : "false");
            fflush(stdout);
            ds4_tokens_free(&prefix);
        }
        ds4_tokens_free(&chat);
        free(prompt);
        free(name);
    }
    fprintf(stderr, "S4.1 capture: %d/%d cases pass\n", passed, cases);
    ds4_session_snapshot_free(&base);
    free(prompts);
    free(row);
    free(evalf);
    ds4_tokens_free(&filler);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return cases > 0 && passed == cases ? 0 : 1;
}

/* The integer after "key": in a JSON line (its first occurrence), or def. */
static long json_int(const char *line, const char *key, long def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *p = strstr(line, pat);
    return p ? strtol(p + strlen(pat), NULL, 10) : def;
}

/* One prompt of llama.cpp's dump (bench/dflash-dump, index.jsonl): its
 * tokens by position, the feature row of each position, and which rows are
 * generated (each one an anchor with a draft). */
typedef struct {
    int prompt, n, n_prompt;
    int *token, *feature_row;
    char name[64];
} dump_prompt;

static int read_dump(const char *index_path, dump_prompt *out, int max) {
    char *text = read_file(index_path);
    int n = 0;
    for (char *line = strtok(text, "\n"); line; line = strtok(NULL, "\n")) {
        const int prompt = (int)json_int(line, "prompt", -1), pos = (int)json_int(line, "pos", -1);
        if (prompt < 0 || pos < 0) die("index line without prompt or pos", line);
        if (n == 0 || out[n - 1].prompt != prompt) {
            if (n == max) die("too many prompts in the index", index_path);
            dump_prompt *d = &out[n++];
            memset(d, 0, sizeof(*d));
            d->prompt = prompt;
            d->token = malloc(65536 * sizeof(int));
            d->feature_row = malloc(65536 * sizeof(int));
            char *name = json_field(line, "name");
            snprintf(d->name, sizeof(d->name), "%s", name ? name : "?");
            free(name);
        }
        dump_prompt *d = &out[n - 1];
        if (pos != d->n || pos >= 65536) die("index positions out of order", line);
        d->token[pos] = (int)json_int(line, "token", -1);
        d->feature_row[pos] = (int)json_int(line, "feature_row", -1);
        if (strstr(line, "\"source\":\"prompt\"")) d->n_prompt = pos + 1;
        d->n++;
    }
    free(text);
    return n;
}

static void read_feature_row(FILE *fp, int row, int width, float *out) {
    if (fseeko(fp, (off_t)row * width * (off_t)sizeof(float), SEEK_SET) != 0 ||
        fread(out, sizeof(float), (size_t)width, fp) != (size_t)width) {
        die("cannot read a feature row", NULL);
    }
}

/* S4.1 against llama.cpp: per prompt of the dump, our session syncs the
 * prompt's ids and evaluates its generated tokens one at a time in DFlash
 * mode; at every position, each tap's cosine with llama.cpp's feature row.
 * One JSON line per position; the summary takes the generated positions
 * (one-row decodes on both sides) of the first --prompts prompts. */
static int run_features(int argc, char **argv) {
    if (argc < 6) die("features needs MODEL DRAFTER INDEX FEATURES", NULL);
    const live_args a = parse_args(argc, argv, 2);
    dump_prompt dp[16];
    const int n_dp = read_dump(a.rest[0], dp, 16);
    FILE *ff = fopen(a.rest[1], "rb");
    if (!ff) die("cannot open", a.rest[1]);
    ds4_engine *engine = open_or_die(&a);
    const int width = ds4_engine_dflash_feature_floats(engine), taps = width / 5120;
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0) die("session create failed", NULL);
    if (ds4_session_set_verify_mode(s, DS4_VERIFY_MODE_MMA) != 0) die("DFlash mode refused", NULL);
    float *ours = malloc((size_t)width * sizeof(float)), *theirs = malloc((size_t)width * sizeof(float));
    double worst[8];
    for (int k = 0; k < 8; k++) worst[k] = 1.0;
    int checked = 0;
    for (int i = a.first; i < n_dp && i < a.first + a.count; i++) {
        const dump_prompt *d = &dp[i];
        ds4_tokens prefix = {0};
        for (int p = 0; p < d->n_prompt; p++) ds4_tokens_push(&prefix, d->token[p]);
        sync_or_die(s, &prefix);
        for (int p = 0; p < d->n; p++) {
            if (p >= d->n_prompt) eval_or_die(s, d->token[p]);
            if (p + 1 < d->n_prompt) continue;
            for (int q = p == d->n_prompt - 1 ? 0 : p; q <= p; q++) {
                features_or_die(s, q, ours);
                read_feature_row(ff, d->feature_row[q], width, theirs);
                printf("{\"prompt\":\"%s\",\"pos\":%d,\"source\":\"%s\",\"cos\":[", d->name, q,
                       q < d->n_prompt ? "prompt" : "generated");
                for (int k = 0; k < taps; k++) {
                    double dot = 0.0, na = 0.0, nb = 0.0, dmax = 0.0, bmax = 0.0;
                    for (int c = 0; c < 5120; c++) {
                        const double x = ours[k * 5120 + c], y = theirs[k * 5120 + c];
                        dot += x * y; na += x * x; nb += y * y;
                        dmax = fmax(dmax, fabs(x - y)); bmax = fmax(bmax, fabs(y));
                    }
                    const double cosv = dot / sqrt(na * nb);
                    printf("%s%.7f", k ? "," : "", cosv);
                    if (q >= d->n_prompt && cosv < worst[k]) worst[k] = cosv;
                }
                printf("]}\n");
            }
            if (p >= d->n_prompt) checked++;
        }
        ds4_tokens_free(&prefix);
    }
    fprintf(stderr, "S4.1 features: %d generated positions; worst cosine per tap:", checked);
    bool pass = checked > 0;
    for (int k = 0; k < taps; k++) {
        fprintf(stderr, " %.7f", worst[k]);
        pass = pass && worst[k] >= 0.999;
    }
    fprintf(stderr, " (pass at >= 0.999: %s)\n", pass ? "yes" : "no");
    free(ours); free(theirs);
    fclose(ff);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return pass ? 0 : 1;
}

/* S4.2 live: one prompt at one depth.  The session syncs the prefix (the
 * sync injects the rows it captured), verifies [x0, 7 filler drafts] and
 * commits 5 rows, which injects them.  For the 16 positions N - 11 .. N + 4
 * it writes each position's captured features and every block's ring K and
 * V to OUT (int32 header: positions, feature floats, blocks, K/V floats;
 * int32 positions; then per position its features and per block K and V,
 * f32), for a double recompute from the drafter's weights.  The 3 rejected
 * rows must have no ring entry. */
static int run_inject(int argc, char **argv) {
    if (argc < 7) die("inject needs MODEL DRAFTER PROMPTS FILLER OUT", NULL);
    live_args a = parse_args(argc, argv, 3);
    ds4_engine *engine = open_or_die(&a);
    const int width = ds4_engine_dflash_feature_floats(engine), n_block = 5, kv = 1024, n_pos = 16;
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0) die("session create failed", NULL);
    if (ds4_session_set_verify_mode(s, DS4_VERIFY_MODE_MMA) != 0) die("DFlash mode refused", NULL);
    char *filler_text = read_file(a.rest[1]);
    ds4_tokens filler = {0};
    ds4_tokenize_text(engine, filler_text, &filler);
    free(filler_text);
    char *prompts = read_file(a.rest[0]);
    char *line = strtok(prompts, "\n");
    for (int i = 0; line && i < a.first; i++) line = strtok(NULL, "\n");
    char *prompt = line ? json_field(line, "prompt") : NULL;
    if (!prompt) die("no prompt", a.rest[0]);
    ds4_tokens chat = {0}, prefix = {0};
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_HIGH, &chat);
    for (int i = 0; i + chat.len < a.depths[0]; i++) ds4_tokens_push(&prefix, filler.v[i % filler.len]);
    for (int i = 0; i < chat.len; i++) ds4_tokens_push(&prefix, chat.v[i]);
    const int n = prefix.len;
    sync_or_die(s, &prefix);
    int drafts[ROWS - 1];
    for (int i = 0; i < ROWS - 1; i++) drafts[i] = filler.v[(n + i) % filler.len];
    char err[256] = {0};
    if (ds4_session_verify(s, ds4_session_argmax(s), drafts, ROWS - 1, err, sizeof(err)) != 0) die("verify failed", err);
    if (ds4_session_verify_commit(s, 5, err, sizeof(err)) != 0) die("commit failed", err);
    float *f = malloc((size_t)width * sizeof(float)), *k = malloc((size_t)kv * sizeof(float));
    float *v = malloc((size_t)kv * sizeof(float));
    int rejected_absent = 0;
    for (int p = n + 5; p < n + ROWS; p++) {
        int absent = 1;
        for (int b = 0; b < n_block; b++) absent &= ds4_session_dflash_ring(s, b, p, k, v) != 0;
        rejected_absent += absent;
    }
    FILE *fp = fopen(a.rest[2], "wb");
    if (!fp) die("cannot write", a.rest[2]);
    const int32_t header[4] = { n_pos, width, n_block, kv };
    fwrite(header, sizeof(header), 1, fp);
    for (int i = 0; i < n_pos; i++) {
        const int32_t p = n - 11 + i;
        fwrite(&p, sizeof(p), 1, fp);
    }
    for (int i = 0; i < n_pos; i++) {
        const int p = n - 11 + i;
        features_or_die(s, p, f);
        fwrite(f, sizeof(float), (size_t)width, fp);
        for (int b = 0; b < n_block; b++) {
            if (ds4_session_dflash_ring(s, b, p, k, v) != 0) die("a committed position has no ring entry", NULL);
            fwrite(k, sizeof(float), (size_t)kv, fp);
            fwrite(v, sizeof(float), (size_t)kv, fp);
        }
    }
    fclose(fp);
    printf("{\"depth\":%d,\"pos0\":%d,\"positions\":[%d,%d],\"committed\":5,\"rejected_rows_absent\":%d}\n",
           a.depths[0], n, n - 11, n + 4, rejected_absent);
    free(f); free(k); free(v);
    ds4_tokens_free(&prefix); ds4_tokens_free(&chat); ds4_tokens_free(&filler);
    free(prompt); free(prompts);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return rejected_absent == ROWS - 5 ? 0 : 1;
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* G7: llama.cpp's features into our drafter.  Per prompt of the dump the
 * session's ring is filled with llama.cpp's feature rows of the prompt
 * (positions 0..N-1); then for each generated position p, our drafter
 * drafts after the dump's token at p from the features of 0..p-1, and
 * llama.cpp's row p is appended, as llama.cpp's dump did.  The target model
 * is loaded for its embeddings and head only; nothing is evaluated.  One
 * JSON line per draft (drafts and draft_ms); the lattice of rows 1..n goes
 * to LATTICE in the dump's layout (per row 16 candidate ids as f32, then
 * scores[j][k]).  The comparison is the s5 data dir's g7_compare.py. */
static int run_g7(int argc, char **argv) {
    if (argc < 7) die("g7 needs MODEL DRAFTER INDEX FEATURES LATTICE", NULL);
    live_args a = parse_args(argc, argv, 3);
    dump_prompt dp[16];
    const int n_dp = read_dump(a.rest[0], dp, 16);
    FILE *ff = fopen(a.rest[1], "rb"), *fl = fopen(a.rest[2], "wb");
    if (!ff || !fl) die("cannot open the features or the lattice", NULL);
    ds4_engine *engine = open_or_die(&a);
    const int width = ds4_engine_dflash_feature_floats(engine), K = ds4_engine_dflash_top_k(engine);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0) die("session create failed", NULL);
    float *rows = malloc((size_t)2048 * width * sizeof(float));
    int *cand = malloc((size_t)ROWS * K * sizeof(int));
    float *scores = malloc((size_t)ROWS * K * K * sizeof(float)), *out = malloc((size_t)(K + K * K) * sizeof(float));
    int drafts_done = 0;
    for (int i = a.first; i < n_dp && i < a.first + a.count; i++) {
        const dump_prompt *d = &dp[i];
        for (int p = 0; p < d->n_prompt; p++) read_feature_row(ff, d->feature_row[p], width, rows + (size_t)p * width);
        if (ds4_session_dflash_set_features(s, 0, d->n_prompt, rows) != 0) die("set features failed", d->name);
        for (int p = d->n_prompt; p < d->n; p++) {
            int draft[ROWS];
            char err[256] = {0};
            const double t0 = now_ms();
            const int n = ds4_session_dflash_draft_at(s, p, d->token[p], ROWS - 1, draft, cand, scores, err, sizeof(err));
            const double ms = now_ms() - t0;
            if (n != ROWS - 1) die("draft failed", err);
            printf("{\"prompt\":%d,\"name\":\"%s\",\"pos\":%d,\"anchor\":%d,\"drafts\":[", d->prompt, d->name, p,
                   d->token[p]);
            for (int j = 0; j < n; j++) printf("%s%d", j ? "," : "", draft[j]);
            printf("],\"draft_ms\":%.3f}\n", ms);
            for (int r = 1; r <= n; r++) {
                for (int k = 0; k < K; k++) out[k] = (float)cand[r * K + k];
                memcpy(out + K, scores + (size_t)r * K * K, (size_t)K * K * sizeof(float));
                fwrite(out, sizeof(float), (size_t)(K + K * K), fl);
            }
            drafts_done++;
            read_feature_row(ff, d->feature_row[p], width, rows);
            if (ds4_session_dflash_set_features(s, p, 1, rows) != 0) die("set features failed", d->name);
        }
        fflush(stdout);
    }
    ds4_spec_stats st;
    ds4_session_spec_stats(s, &st);
    fprintf(stderr, "G7: %d drafts, draft_ms total %.1f (mean %.2f)\n", drafts_done, st.draft_ms,
            drafts_done ? st.draft_ms / drafts_done : 0.0);
    fclose(fl); fclose(ff);
    free(rows); free(cand); free(scores); free(out);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return drafts_done > 0 ? 0 : 1;
}

/* The product path once, greedy: per prompt at one depth, a DFlash-mode
 * session decodes --tokens tokens serially (the reference), then, from the
 * same snapshot, speculatively: draft after the anchor, verify, select rows
 * while each row's argmax equals its draft, commit, and the last selected
 * row's argmax is the next anchor.  The two streams must be equal.  One
 * JSON line per step (drafted, accepted, draft/verify/commit ms) and one
 * per prompt. */
static int run_spec(int argc, char **argv) {
    if (argc < 6) die("spec needs MODEL DRAFTER PROMPTS FILLER", NULL);
    live_args a = parse_args(argc, argv, 2);
    const int tokens = a.tokens;
    ds4_engine *engine = open_or_die(&a);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0) die("session create failed", NULL);
    if (ds4_session_set_verify_mode(s, DS4_VERIFY_MODE_MMA) != 0) die("DFlash mode refused", NULL);
    char *filler_text = read_file(a.rest[1]);
    ds4_tokens filler = {0};
    ds4_tokenize_text(engine, filler_text, &filler);
    free(filler_text);
    char *prompts = read_file(a.rest[0]);
    int index = 0, cases = 0, passed = 0;
    int *ref = malloc((size_t)(tokens + ROWS) * sizeof(int)), *got = malloc((size_t)(tokens + ROWS) * sizeof(int));
    ds4_session_snapshot base = {0};
    for (char *line = strtok(prompts, "\n"); line; line = strtok(NULL, "\n"), index++) {
        if (index < a.first || index >= a.first + a.count) continue;
        char *name = json_field(line, "name"), *prompt = json_field(line, "prompt");
        ds4_tokens chat = {0}, prefix = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_HIGH, &chat);
        for (int i = 0; i + chat.len < a.depths[0]; i++) ds4_tokens_push(&prefix, filler.v[i % filler.len]);
        for (int i = 0; i < chat.len; i++) ds4_tokens_push(&prefix, chat.v[i]);
        if (prefix.len + tokens + ROWS > a.ctx) die("prefix does not fit the context", name);
        sync_or_die(s, &prefix);
        char err[256] = {0};
        if (ds4_session_save_snapshot(s, &base, err, sizeof(err)) != 0) die("snapshot save failed", err);
        double serial_ms = now_ms();
        for (int i = 0; i < tokens; i++) {
            ref[i] = ds4_session_argmax(s);
            if (i + 1 < tokens) eval_or_die(s, ref[i]);
        }
        serial_ms = now_ms() - serial_ms;
        if (ds4_session_load_snapshot(s, &base, err, sizeof(err)) != 0) die("snapshot load failed", err);
        ds4_spec_stats st0, st1;
        ds4_session_spec_stats(s, &st0);
        int n_got = 0, steps = 0, drafted = 0, accepted = 0;
        int anchor = ds4_session_argmax(s);
        double spec_ms = now_ms();
        while (n_got < tokens) {
            int draft[ROWS];
            ds4_spec_stats a0, a1;
            ds4_session_spec_stats(s, &a0);
            int n = ds4_session_dflash_draft(s, anchor, ROWS - 1, draft, err, sizeof(err));
            if (n < 0) die("draft failed", err);
            if (n > tokens - n_got - 1) n = tokens - n_got - 1;
            if (ds4_session_verify(s, anchor, draft, n, err, sizeof(err)) != 0) die("verify failed", err);
            int k = 0, next = -1;
            for (int r = 0; r <= n; r++) {
                if (ds4_session_verify_select(s, r) != 0) die("select refused", NULL);
                next = ds4_session_argmax(s);
                if (r == n || next != draft[r]) { k = r; break; }
            }
            if (ds4_session_verify_commit(s, k + 1, err, sizeof(err)) != 0) die("commit failed", err);
            ds4_session_spec_stats(s, &a1);
            got[n_got++] = anchor;
            for (int r = 0; r < k; r++) got[n_got++] = draft[r];
            printf("{\"prompt\":\"%s\",\"step\":%d,\"pos\":%d,\"drafted\":%d,\"accepted\":%d,\"draft_ms\":%.3f,"
                   "\"verify_ms\":%.3f,\"commit_ms\":%.3f}\n", name, steps, ds4_session_pos(s) - k - 1, n, k,
                   a1.draft_ms - a0.draft_ms, a1.verify_ms - a0.verify_ms, a1.commit_ms - a0.commit_ms);
            steps++; drafted += n; accepted += k;
            anchor = next;
        }
        spec_ms = now_ms() - spec_ms;
        ds4_session_spec_stats(s, &st1);
        int equal = 0;
        while (equal < tokens && got[equal] == ref[equal]) equal++;
        const bool pass = equal == tokens;
        cases++;
        passed += pass;
        printf("{\"prompt\":\"%s\",\"depth\":%d,\"pos0\":%d,\"tokens\":%d,\"stream_equal_prefix\":%d,\"steps\":%d,"
               "\"drafted\":%d,\"accepted\":%d,\"serial_ms\":%.1f,\"spec_ms\":%.1f,\"draft_ms\":%.1f,"
               "\"verify_ms\":%.1f,\"commit_ms\":%.1f,\"pass\":%s}\n",
               name, a.depths[0], prefix.len, tokens, equal, steps, drafted, accepted, serial_ms, spec_ms,
               st1.draft_ms - st0.draft_ms, st1.verify_ms - st0.verify_ms, st1.commit_ms - st0.commit_ms,
               pass ? "true" : "false");
        fflush(stdout);
        ds4_tokens_free(&prefix); ds4_tokens_free(&chat);
        free(name); free(prompt);
    }
    fprintf(stderr, "spec: %d/%d streams equal\n", passed, cases);
    ds4_session_snapshot_free(&base);
    free(ref); free(got); free(prompts);
    ds4_tokens_free(&filler);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return cases > 0 && passed == cases ? 0 : 1;
}

#define RING_BLOCKS 8    /* DS4_DFLASH_MAX_BLOCKS: blocks past the drafter's read as absent */
#define RING_KV 1024     /* the drafter's 8 K/V heads of 128 */
#define RING_SLOTS 2048
#define PAYLOAD_EVALS 5  /* greedy evals before the save: rows a save must inject */

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/* FNV-1a-64 over every block's ring entries for the window of positions
 * before pos (an absent entry as a zero byte, a present one as a one byte
 * and its K and V floats); *present counts the present entries */
static uint64_t ring_digest(ds4_session *s, int pos, int *present) {
    static float k[RING_KV], v[RING_KV];
    uint64_t h = 1469598103934665603ull;
    *present = 0;
    for (int b = 0; b < RING_BLOCKS; b++) {
        for (int p = pos > RING_SLOTS ? pos - RING_SLOTS : 0; p < pos; p++) {
            const unsigned char has = ds4_session_dflash_ring(s, b, p, k, v) == 0;
            h = fnv1a(h, &has, 1);
            if (!has) continue;
            (*present)++;
            h = fnv1a(h, k, sizeof(k));
            h = fnv1a(h, v, sizeof(v));
        }
    }
    return h;
}

/* a draft at the session's position after its argmax; drafts and an
 * FNV-1a-64 of the lattice of rows 1..n (K candidates per row) */
static int draft_digest(ds4_session *s, int K, int *draft, uint64_t *lattice) {
    int *cand = malloc((size_t)ROWS * K * sizeof(int));
    float *scores = malloc((size_t)ROWS * K * K * sizeof(float));
    char err[256] = {0};
    const int n = ds4_session_dflash_draft_at(s, ds4_session_pos(s), ds4_session_argmax(s), ROWS - 1, draft, cand,
                                              scores, err, sizeof(err));
    if (n < 0) die("draft failed", err);
    *lattice = fnv1a(fnv1a(1469598103934665603ull, cand + K, (size_t)n * K * sizeof(int)),
                     scores + (size_t)K * K, (size_t)n * K * K * sizeof(float));
    free(cand); free(scores);
    return n;
}

static void print_ints(const char *key, const int *v, int n) {
    printf(",\"%s\":[", key);
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", v[i]);
    printf("]");
}

/* the ring and a draft as JSON fields, when the session has a drafter */
static void print_ring_and_draft(ds4_session *s, int K) {
    int present = 0, draft[ROWS];
    uint64_t lattice = 0;
    const uint64_t ring = ring_digest(s, ds4_session_pos(s), &present);
    const int n = draft_digest(s, K, draft, &lattice);
    printf(",\"ring\":\"%016llx\",\"present\":%d", (unsigned long long)ring, present);
    print_ints("drafts", draft, n);
    printf(",\"lattice\":\"%016llx\"", (unsigned long long)lattice);
}

/* tokens greedy tokens, one eval each: the tokens and each step's logits
 * as an FNV-1a-64, as one JSON line */
static void print_continuation(ds4_session *s, const char *label, int tokens) {
    static float logits[1 << 18];
    printf("{\"case\":\"%s\",\"pos\":%d,\"tokens\":[", label, ds4_session_pos(s));
    uint64_t *h = malloc((size_t)tokens * sizeof(uint64_t));
    for (int i = 0; i < tokens; i++) {
        const int n = ds4_session_copy_logits(s, logits, 1 << 18);
        if (n <= 0) die("no logits", NULL);
        h[i] = fnv1a(1469598103934665603ull, logits, (size_t)n * sizeof(float));
        const int t = ds4_session_argmax(s);
        printf("%s%d", i ? "," : "", t);
        eval_or_die(s, t);
    }
    printf("],\"logits\":[");
    for (int i = 0; i < tokens; i++) printf("%s\"%016llx\"", i ? "," : "", (unsigned long long)h[i]);
    printf("]}\n");
    fflush(stdout);
    free(h);
}

static uint64_t save_payload_or_die(ds4_session *s, const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) die("cannot open", path);
    char err[256] = {0};
    if (ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) die("payload save failed", err);
    const long n = ftell(fp);
    if (fclose(fp) != 0 || n < 0) die("cannot write", path);
    return (uint64_t)n;
}

/* S6, the payload trailer (spec section 3.8).  payload-save: a session with
 * the drafter syncs FILLER's tokens and prompt --first's chat to the depth,
 * evaluates PAYLOAD_EVALS greedy tokens one at a time (rows the drafter has
 * not injected yet) and saves its payload to OUT.  A snapshot of it loads
 * into a second session, whose ring must equal the first's after the
 * first's draft has injected anything still pending (presence and K/V bytes
 * of every block over the last window), and whose draft at the same anchor
 * must equal the first's (drafts and lattice).  Then the first session
 * decodes --tokens greedy tokens.  payload-load: a session (with the
 * drafter, or none when DRAFTER is "-") loads IN, reports its ring and a
 * draft when it has a drafter, saves its own payload to OUT unless OUT is
 * "-", and decodes --tokens greedy tokens.  The verdicts across runs are
 * payload_check.py's. */
static int run_payload_save(int argc, char **argv) {
    if (argc < 7) die("payload-save needs MODEL DRAFTER PROMPTS FILLER OUT", NULL);
    live_args a = parse_args(argc, argv, 3);
    ds4_engine *engine = open_or_die(&a);
    ds4_session *s = NULL, *t = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0 || ds4_session_create(&t, engine, a.ctx) != 0) {
        die("session create failed", NULL);
    }
    char *filler_text = read_file(a.rest[1]);
    ds4_tokens filler = {0};
    ds4_tokenize_text(engine, filler_text, &filler);
    free(filler_text);
    char *prompts = read_file(a.rest[0]), *line = strtok(prompts, "\n");
    for (int i = 0; line && i < a.first; i++) line = strtok(NULL, "\n");
    if (!line) die("no such prompt", NULL);
    char *name = json_field(line, "name"), *prompt = json_field(line, "prompt");
    ds4_tokens chat = {0}, prefix = {0};
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_HIGH, &chat);
    for (int i = 0; i + chat.len < a.depths[0]; i++) ds4_tokens_push(&prefix, filler.v[i % filler.len]);
    for (int i = 0; i < chat.len; i++) ds4_tokens_push(&prefix, chat.v[i]);
    if (prefix.len + PAYLOAD_EVALS + a.tokens + ROWS > a.ctx) die("prefix does not fit the context", name);
    sync_or_die(s, &prefix);
    for (int i = 0; i < PAYLOAD_EVALS; i++) eval_or_die(s, ds4_session_argmax(s));
    const uint64_t bytes = save_payload_or_die(s, a.rest[2]);
    printf("{\"case\":\"save\",\"prompt\":\"%s\",\"depth\":%d,\"pos\":%d,\"bytes\":%llu}\n", name, a.depths[0],
           ds4_session_pos(s), (unsigned long long)bytes);
    ds4_session_snapshot snap = {0};
    char err[256] = {0};
    if (ds4_session_save_snapshot(s, &snap, err, sizeof(err)) != 0) die("snapshot save failed", err);
    if (ds4_session_load_snapshot(t, &snap, err, sizeof(err)) != 0) die("snapshot load failed", err);
    int present_t = 0;
    const uint64_t ring_t = ring_digest(t, ds4_session_pos(t), &present_t);
    int draft_s[ROWS], draft_t[ROWS];
    uint64_t lat_s = 0, lat_t = 0;
    const int K = ds4_engine_dflash_top_k(engine);
    const int n_s = draft_digest(s, K, draft_s, &lat_s);
    int present_s = 0;
    const uint64_t ring_s = ring_digest(s, ds4_session_pos(s), &present_s);
    const int n_t = draft_digest(t, K, draft_t, &lat_t);
    const bool pass = ring_s == ring_t && present_s == present_t && n_s == n_t &&
                      !memcmp(draft_s, draft_t, (size_t)n_s * sizeof(int)) && lat_s == lat_t;
    printf("{\"case\":\"roundtrip\",\"snapshot_bytes\":%llu,\"ring\":\"%016llx\",\"present\":%d,"
           "\"ring_loaded\":\"%016llx\",\"present_loaded\":%d",
           (unsigned long long)snap.len, (unsigned long long)ring_s, present_s, (unsigned long long)ring_t,
           present_t);
    print_ints("drafts", draft_s, n_s);
    print_ints("drafts_loaded", draft_t, n_t);
    printf(",\"lattice\":\"%016llx\",\"lattice_loaded\":\"%016llx\",\"pass\":%s}\n", (unsigned long long)lat_s,
           (unsigned long long)lat_t, pass ? "true" : "false");
    print_continuation(s, "continue", a.tokens);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&prefix); ds4_tokens_free(&chat); ds4_tokens_free(&filler);
    free(name); free(prompt); free(prompts);
    ds4_session_free(t);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return pass ? 0 : 1;
}

static int run_payload_load(int argc, char **argv) {
    if (argc < 6) die("payload-load needs MODEL DRAFTER|- IN OUT|-", NULL);
    live_args a = parse_args(argc, argv, 2);
    ds4_engine *engine = open_or_die(&a);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, a.ctx) != 0) die("session create failed", NULL);
    FILE *fp = fopen(a.rest[0], "rb");
    if (!fp || fseek(fp, 0, SEEK_END) != 0) die("cannot open", a.rest[0]);
    const long bytes = ftell(fp);
    char err[256] = {0};
    if (bytes <= 0 || fseek(fp, 0, SEEK_SET) != 0) die("cannot size", a.rest[0]);
    if (ds4_session_load_payload(s, fp, (uint64_t)bytes, err, sizeof(err)) != 0) die("payload load failed", err);
    fclose(fp);
    const bool drafter = strcmp(a.drafter, "-") != 0;
    printf("{\"case\":\"load\",\"drafter\":%s,\"in_bytes\":%ld,\"pos\":%d", drafter ? "true" : "false", bytes,
           ds4_session_pos(s));
    if (drafter) print_ring_and_draft(s, ds4_engine_dflash_top_k(engine));
    if (strcmp(a.rest[1], "-")) printf(",\"out_bytes\":%llu", (unsigned long long)save_payload_or_die(s, a.rest[1]));
    printf("}\n");
    print_continuation(s, "continue", a.tokens);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s capture MODEL DRAFTER PROMPTS FILLER [options]\n"
                        "       %s features MODEL DRAFTER INDEX FEATURES [--prompts N] [--first I] [--ctx N]\n",
                argv[0], argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "capture")) return run_capture(argc, argv);
    if (!strcmp(argv[1], "features")) return run_features(argc, argv);
    if (!strcmp(argv[1], "inject")) return run_inject(argc, argv);
    if (!strcmp(argv[1], "g7")) return run_g7(argc, argv);
    if (!strcmp(argv[1], "spec")) return run_spec(argc, argv);
    if (!strcmp(argv[1], "payload-save")) return run_payload_save(argc, argv);
    if (!strcmp(argv[1], "payload-load")) return run_payload_load(argc, argv);
    die("unknown command", argv[1]);
    return 2;
}
