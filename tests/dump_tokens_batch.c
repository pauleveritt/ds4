/* dump_tokens_batch — print the engine's *content* token ids for many
 * strings on one model open, so a reference tokenizer (the model's own
 * tokenizer.json, with added tokens emptied) can be diffed against the
 * engine over a fuzz corpus.  This is the engine side of P21's Q2 leg.
 *
 *   tests/dump_tokens_batch MODEL.gguf [CORPUS]
 *
 * CORPUS defaults to stdin.  It is a sequence of length-prefixed records —
 * a decimal byte count, a newline, then exactly that many bytes — repeated
 * with no separator.  No byte value is reserved, so a record may hold
 * newlines, CRs, invalid UTF-8 or control-token spellings; the one refusal is
 * an embedded NUL, which the C string tokenizer interface cannot carry.
 *
 * stdout carries one machine-readable line per record: the record's index
 * from 0, then its token ids, space-separated.  Everything else goes to
 * stderr.  A malformed record exits 1; bad arguments exit 2.
 *
 * Unlike tests/dump_chat_transcript this drives no chat rendering at all: it
 * calls the same bpe_tokenize_text a Mellum turn's *content* goes through
 * (mellum_chat_append_wrapped), never the special-aware
 * tokenize_rendered_chat_vocab that `--dump-tokens` uses.  Keeping it a
 * separate binary leaves P20's G4 instrument undisturbed.
 *
 * The model is opened for inspection only — no GPU, no generation, never
 * --raw-prompt.
 */

#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || !strncmp(argv[1], "--", 2)) {
        fprintf(stderr, "usage: dump_tokens_batch MODEL.gguf [CORPUS]\n");
        return 2;
    }
    const char *model_path = argv[1];

    FILE *in = stdin;
    if (argc == 3) {
        in = fopen(argv[2], "rb");
        if (!in) {
            fprintf(stderr, "dump_tokens_batch: cannot read %s\n", argv[2]);
            return 2;
        }
    }

    const int rc = ds4_dump_text_tokenization_batch(model_path, in, stdout);
    if (in != stdin) fclose(in);
    if (fflush(stdout) != 0) {
        fprintf(stderr, "dump_tokens_batch: cannot write stdout\n");
        return 1;
    }
    if (rc != 0) {
        fprintf(stderr, "dump_tokens_batch: malformed corpus (rc=%d)\n", rc);
        return 1;
    }
    return 0;
}
