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
 * Usage: test_dflash_live capture MODEL DRAFTER PROMPTS FILLER [--prompts N]
 *                                 [--depths D,D,...] [--ctx N]
 *        test_dflash_live features MODEL DRAFTER INDEX FEATURES [--prompts N]
 *                                 [--first I] [--ctx N]
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
    int ctx, first, count, n_depths, depths[8];
    char **rest;
    int n_rest;
} live_args;

static live_args parse_args(int argc, char **argv, int n_fixed) {
    live_args a = { .model = argv[2], .drafter = argv[3], .ctx = 10240, .count = 1 << 30,
                    .n_depths = 3, .depths = {100, 2600, 9700} };
    a.rest = argv + 4;
    a.n_rest = n_fixed;
    for (int i = 4 + n_fixed; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--prompts")) a.count = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--first")) a.first = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--ctx")) a.ctx = atoi(argv[i + 1]);
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

static ds4_engine *open_or_die(const live_args *a) {
    ds4_engine_options opt = {.model_path = a->model, .mtp_path = a->drafter, .context_size = a->ctx,
                              .backend = DS4_BACKEND_METAL};
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("engine open failed", a->model);
    if (!ds4_engine_has_dflash(engine)) die("the engine has no drafter", a->drafter);
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s capture MODEL DRAFTER PROMPTS FILLER [options]\n"
                        "       %s features MODEL DRAFTER INDEX FEATURES [--prompts N] [--first I] [--ctx N]\n",
                argv[0], argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "capture")) return run_capture(argc, argv);
    if (!strcmp(argv[1], "features")) return run_features(argc, argv);
    die("unknown command", argv[1]);
    return 2;
}
