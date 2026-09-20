/* dump_chat_transcript — print the engine's token ids for a whole chat
 * transcript, so a reference renderer (the model's own chat_template.jinja
 * plus its tokenizer.json) can be diffed against the engine's hand-written
 * chat rendering for roles `--dump-chat-tokens` cannot reach: assistant and
 * tool.  This is the engine side of P20's G4 leg.
 *
 *   tests/dump_chat_transcript [--think] MODEL.gguf TRANSCRIPT
 *
 * TRANSCRIPT is UTF-8.  Each record is the role name, a newline, then the
 * content, then the record separator byte 0x1E (so content may contain
 * newlines).  A trailing separator is optional.  Roles are passed through to
 * ds4_chat_append_message verbatim (system, user, assistant, tool, ...).
 *
 * The model is opened for inspection only — no GPU, no generation, never
 * --raw-prompt.  A generation prompt (ds4_chat_append_assistant_prefix) is
 * always appended, matching the reference renderer's
 * add_generation_prompt=True; --think selects thinking mode (the default is
 * non-thinking, i.e. what ds4's --nothink renders).
 *
 * stdout carries exactly one line: the space-separated token ids.  Everything
 * else goes to stderr.
 */

#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RECORD_SEPARATOR '\x1e'

static char *read_whole_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { fclose(fp); return NULL; }
    for (;;) {
        if (len == cap) {
            char *grown = realloc(buf, cap * 2);
            if (!grown) { free(buf); fclose(fp); return NULL; }
            buf = grown;
            cap *= 2;
        }
        const size_t n = fread(buf + len, 1, cap - len, fp);
        len += n;
        if (n == 0) break;
    }
    const int bad = ferror(fp);
    fclose(fp);
    if (bad) { free(buf); return NULL; }
    char *grown = realloc(buf, len + 1);
    if (!grown) { free(buf); return NULL; }
    grown[len] = '\0';
    *len_out = len;
    return grown;
}

/* Split the buffer in place.  Returns the record count, or -1 on a malformed
 * record (one with no newline separating role from content). */
static int split_records(char *buf, size_t len, char ***roles_out, char ***contents_out) {
    size_t count = 0;
    for (size_t i = 0; i < len; i++) {
        if (buf[i] == RECORD_SEPARATOR) count++;
    }
    if (len && buf[len - 1] != RECORD_SEPARATOR) count++;   /* trailing record */
    if (!count) { *roles_out = NULL; *contents_out = NULL; return 0; }

    char **roles = calloc(count, sizeof(*roles));
    char **contents = calloc(count, sizeof(*contents));
    if (!roles || !contents) { free(roles); free(contents); return -1; }

    size_t n = 0;
    char *p = buf;
    char *end = buf + len;
    while (p < end && n < count) {
        char *sep = memchr(p, RECORD_SEPARATOR, (size_t)(end - p));
        char *record_end = sep ? sep : end;
        char *nl = memchr(p, '\n', (size_t)(record_end - p));
        if (!nl) {
            fprintf(stderr, "dump_chat_transcript: record %zu has no role line\n", n);
            free(roles); free(contents);
            return -1;
        }
        *nl = '\0';
        *record_end = '\0';
        roles[n] = p;
        contents[n] = nl + 1;
        n++;
        p = record_end + 1;
    }

    *roles_out = roles;
    *contents_out = contents;
    return (int)n;
}

int main(int argc, char **argv) {
    bool thinking = false;
    int arg = 1;
    while (arg < argc && !strncmp(argv[arg], "--", 2)) {
        if (!strcmp(argv[arg], "--think")) {
            thinking = true;
            arg++;
        } else {
            fprintf(stderr, "dump_chat_transcript: unknown option %s\n", argv[arg]);
            return 2;
        }
    }
    if (argc - arg != 2) {
        fprintf(stderr, "usage: dump_chat_transcript [--think] MODEL.gguf TRANSCRIPT\n");
        return 2;
    }
    const char *model_path = argv[arg];
    const char *transcript_path = argv[arg + 1];

    size_t len = 0;
    char *buf = read_whole_file(transcript_path, &len);
    if (!buf) {
        fprintf(stderr, "dump_chat_transcript: cannot read %s\n", transcript_path);
        return 2;
    }

    char **roles = NULL;
    char **contents = NULL;
    const int count = split_records(buf, len, &roles, &contents);
    if (count < 0) { free(buf); return 2; }

    /* ds4_dump_chat_transcript_tokenization prints the id list plus one
     * "id  token" line per token; only the first line is the id list, and it
     * is comma-separated inside brackets.  Capture it and re-print the ids
     * space-separated, so stdout is exactly one machine-readable line. */
    char *text = NULL;
    size_t text_len = 0;
    FILE *fp = open_memstream(&text, &text_len);
    if (!fp) { free(roles); free(contents); free(buf); return 2; }

    const int rc = ds4_dump_chat_transcript_tokenization(
        model_path,
        (const char *const *)roles,
        (const char *const *)contents,
        (size_t)count,
        thinking ? DS4_THINK_HIGH : DS4_THINK_NONE,
        true,
        fp);
    const int closed = fclose(fp);
    free(roles);
    free(contents);
    free(buf);
    if (rc != 0 || closed != 0 || !text) {
        fprintf(stderr, "dump_chat_transcript: tokenization failed (rc=%d)\n", rc);
        free(text);
        return 1;
    }

    char *newline = strchr(text, '\n');
    if (newline) *newline = '\0';
    char *ids = text;
    const size_t ids_len = strlen(ids);
    if (ids_len < 2 || ids[0] != '[' || ids[ids_len - 1] != ']') {
        fprintf(stderr, "dump_chat_transcript: unexpected id line %s\n", ids);
        free(text);
        return 1;
    }
    ids[ids_len - 1] = '\0';
    ids++;

    bool first = true;
    for (char *tok = strtok(ids, ", "); tok; tok = strtok(NULL, ", ")) {
        if (!first) fputc(' ', stdout);
        fputs(tok, stdout);
        first = false;
    }
    fputc('\n', stdout);
    free(text);
    return 0;
}
