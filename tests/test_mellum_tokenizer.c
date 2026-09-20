#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *dump_chat_first_line(const char *model_path, ds4_think_mode think_mode) {
    char *text = NULL;
    size_t text_len = 0;
    FILE *fp = open_memstream(&text, &text_len);
    if (!fp) return NULL;

    const int rc = ds4_dump_chat_tokenization(
        model_path,
        NULL,
        "Answer with one word: the capital of France is",
        think_mode,
        8192,   /* ctx_size: only consulted for DS4_THINK_MAX (ds4.c:63410) */
        fp);
    if (fclose(fp) != 0 || rc != 0 || !text) {
        free(text);
        return NULL;
    }

    char *newline = strchr(text, '\n');
    if (!newline) {
        free(text);
        return NULL;
    }
    *newline = '\0';
    return text;
}

static char *dump_rendered_first_line(const char *model_path) {
    static const char rendered[] =
        "<|im_start|>user\n"
        "Answer with one word: the capital of France is<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    char *text = NULL;
    size_t text_len = 0;
    FILE *fp = open_memstream(&text, &text_len);
    if (!fp) return NULL;

    const int rc = ds4_dump_text_tokenization(model_path, rendered, fp);
    if (fclose(fp) != 0 || rc != 0 || !text) {
        free(text);
        return NULL;
    }
    char *newline = strchr(text, '\n');
    if (!newline) {
        free(text);
        return NULL;
    }
    *newline = '\0';
    return text;
}

/* ---------------------------------------------------------------------
 * Tool-token cases (P20 Task 3, spec Scope 3).
 *
 * The snapshot's tokenizer.json lists four added tokens that the model's own
 * chat template emits:
 *
 *     29 <tool_call>   30 </tool_call>   31 <tool_response>   32 </tool_response>
 *
 * (recompute: read `added_tokens` out of the snapshot's tokenizer.json; the
 * ids are asserted literally here because the C API exposes no
 * text-to-id lookup).  Before the P20 fix the Mellum vocabulary hard-set all
 * four to -1 and the tool branch used the non-special-aware tokenizer, so the
 * engine rendered `<tool_response>` as `62 3757 97 3309 64`
 * ('<' 'tool' '_' 'response' '>') — P19 finding, F5.  Token 3757 is the
 * standalone word "tool"; its presence is the signature of the split.
 */

#define TOKEN_TOOL_CALL_START   29
#define TOKEN_TOOL_CALL_END     30
#define TOKEN_TOOL_RESP_START   31
#define TOKEN_TOOL_RESP_END     32
#define TOKEN_WORD_TOOL       3757

static char *dump_transcript_first_line(const char *model_path,
                                        const char *const *roles,
                                        const char *const *contents,
                                        size_t count) {
    char *text = NULL;
    size_t text_len = 0;
    FILE *fp = open_memstream(&text, &text_len);
    if (!fp) return NULL;

    const int rc = ds4_dump_chat_transcript_tokenization(
        model_path, roles, contents, count, DS4_THINK_NONE, true, fp);
    if (fclose(fp) != 0 || rc != 0 || !text) {
        free(text);
        return NULL;
    }
    char *newline = strchr(text, '\n');
    if (!newline) {
        free(text);
        return NULL;
    }
    *newline = '\0';
    return text;
}

/* Count occurrences of `id` in a "[a, b, c]" id line. */
static int id_count(const char *line, int id) {
    if (!line) return -1;
    int seen = 0;
    for (const char *p = line; *p; p++) {
        if (*p != '-' && (*p < '0' || *p > '9')) continue;
        const char *start = p;
        while (*p == '-' || (*p >= '0' && *p <= '9')) p++;
        if (atoi(start) == id) seen++;
        if (!*p) break;
    }
    return seen;
}

static bool check(const char *name, bool ok, const char *line) {
    if (ok) {
        fprintf(stderr, "ok   %s\n", name);
    } else {
        fprintf(stderr, "FAIL %s\n       %s\n", name, line ? line : "(no output)");
    }
    return ok;
}

int main(void) {
    const char *model_path = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model_path || !model_path[0]) {
        fprintf(stderr, "DS4_TEST_MELLUM_MODEL must name the Mellum Q8 GGUF\n");
        return 2;
    }

    static const char no_think_expected[] =
        "[27, 1397, 233, 5698, 434, 768, 1574, 60, 302, 5782, 332, 8438, "
        "359, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233]";
    static const char think_expected[] =
        "[27, 1397, 233, 5698, 434, 768, 1574, 60, 302, 5782, 332, 8438, "
        "359, 28, 233, 27, 8091, 233]";

    char *no_think = dump_chat_first_line(model_path, DS4_THINK_NONE);
    char *think = dump_chat_first_line(model_path, DS4_THINK_HIGH);
    char *rendered = dump_rendered_first_line(model_path);
    bool ok = no_think && think && rendered &&
        !strcmp(no_think, no_think_expected) &&
        !strcmp(think, think_expected) &&
        !strcmp(rendered, no_think_expected);
    if (!ok) {
        fprintf(stderr, "FAIL Mellum chat-token fixture mismatch\n");
        if (no_think) fprintf(stderr, "  no-think: %s\n", no_think);
        if (think) fprintf(stderr, "  think:    %s\n", think);
        if (rendered) fprintf(stderr, "  rendered: %s\n", rendered);
    } else {
        fprintf(stderr, "ok   existing chat/rendered fixtures\n");
    }
    free(rendered);
    free(think);
    free(no_think);

    /* (a) An assistant turn whose content carries a Hermes-form tool call.
     * The assistant branch already uses the special-aware tokenizer, so this
     * is purely about the four ids being resolved. */
    {
        const char *roles[] = {"user", "assistant"};
        const char *contents[] = {
            "Read a.py",
            "<tool_call>\n{\"name\": \"read_file\", \"arguments\": "
            "{\"path\": \"a.py\"}}\n</tool_call>",
        };
        char *line = dump_transcript_first_line(model_path, roles, contents, 2);
        ok &= check("assistant turn emits <tool_call>/</tool_call> as 29/30",
                    id_count(line, TOKEN_TOOL_CALL_START) == 1 &&
                    id_count(line, TOKEN_TOOL_CALL_END) == 1 &&
                    id_count(line, TOKEN_WORD_TOOL) == 0,
                    line);
        free(line);
    }

    /* (b) A `tool` role message — the engine's own wrapper, the path
     * ds4_agent.c takes for every tool observation. */
    {
        const char *roles[] = {"user", "tool"};
        const char *contents[] = {"Read a.py", "print('hi')"};
        char *line = dump_transcript_first_line(model_path, roles, contents, 2);
        ok &= check("tool turn wraps content in 31/32",
                    id_count(line, TOKEN_TOOL_RESP_START) == 1 &&
                    id_count(line, TOKEN_TOOL_RESP_END) == 1 &&
                    id_count(line, TOKEN_WORD_TOOL) == 0,
                    line);
        free(line);
    }

    /* (c) Negative control: spellings that are near-misses must stay text.
     * Run through the assistant branch, which IS special-aware, so the check
     * is about special_token_at's exact match and not about the branch. */
    {
        const char *roles[] = {"assistant"};
        const char *contents[] = {
            "<tool_x> <tool_calls> <tool_call and </tool_ > <tool_response",
        };
        char *line = dump_transcript_first_line(model_path, roles, contents, 1);
        ok &= check("near-miss spellings do not become tool tokens",
                    line &&
                    id_count(line, TOKEN_TOOL_CALL_START) == 0 &&
                    id_count(line, TOKEN_TOOL_CALL_END) == 0 &&
                    id_count(line, TOKEN_TOOL_RESP_START) == 0 &&
                    id_count(line, TOKEN_TOOL_RESP_END) == 0,
                    line);
        free(line);
    }

    return ok ? 0 : 1;
}
