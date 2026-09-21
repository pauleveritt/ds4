/* dump_segments_batch — print the Mellum *content* segmenter's piece byte
 * lengths for many strings in one process, so the snapshot's own
 * pre-tokenizer can be diffed against the engine over every Unicode
 * codepoint.  This is the engine side of P21.2's R1 leg.
 *
 *   tests/dump_segments_batch [CORPUS]
 *
 * **No model and no vocabulary are involved.**  mellum_segment_pieces — the
 * segmentation a Mellum turn's content goes through in bpe_tokenize_text, and
 * the one ds4_mellum_segment_content_probe probes — takes bytes and nothing
 * else, so the boundary decision is reachable without opening a GGUF.  That
 * is what makes an exhaustive sweep affordable, and it is why this binary is
 * fast tier: `make test-dump-segments-batch` is ungated.
 *
 * CORPUS defaults to stdin.  It is the record format tests/dump_tokens_batch
 * reads — a decimal byte count, a newline, then exactly that many bytes,
 * repeated with no separator.  No byte value is reserved, so a record may
 * hold newlines, CRs, 0x1f or invalid UTF-8; the one refusal is an embedded
 * NUL, which the C string interface cannot carry.
 *
 * stdout carries one machine-readable line per record: the record's index
 * from 0, the piece count, then each piece's length in bytes.  Lengths rather
 * than the pieces themselves because a piece may hold any byte at all and the
 * caller already holds the record it sent.  Everything else goes to stderr.
 * A malformed record exits 1; bad arguments or an unreadable corpus exit 2.
 */

#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && !strncmp(argv[1], "--", 2))) {
        fprintf(stderr, "usage: dump_segments_batch [CORPUS]\n");
        return 2;
    }

    FILE *in = stdin;
    if (argc == 2) {
        in = fopen(argv[1], "rb");
        if (!in) {
            fprintf(stderr, "dump_segments_batch: cannot read %s\n", argv[1]);
            return 2;
        }
    }

    const int rc = ds4_dump_mellum_segments_batch(in, stdout);
    if (in != stdin) fclose(in);
    if (fflush(stdout) != 0) {
        fprintf(stderr, "dump_segments_batch: cannot write stdout\n");
        return 1;
    }
    if (rc != 0) {
        fprintf(stderr, "dump_segments_batch: malformed corpus (rc=%d)\n", rc);
        return 1;
    }
    return 0;
}
