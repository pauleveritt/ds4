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
    const int ok = no_think && think && rendered &&
        !strcmp(no_think, no_think_expected) &&
        !strcmp(think, think_expected) &&
        !strcmp(rendered, no_think_expected);
    if (!ok) {
        fprintf(stderr, "Mellum chat-token fixture mismatch\n");
        if (no_think) fprintf(stderr, "  no-think: %s\n", no_think);
        if (think) fprintf(stderr, "  think:    %s\n", think);
        if (rendered) fprintf(stderr, "  rendered: %s\n", rendered);
    }
    free(rendered);
    free(think);
    free(no_think);
    return ok ? 0 : 1;
}
