/* Live open of a dense qwen35 target with an optional DFlash drafter: the
 * engine prints the drafter, its plan and the memory line at open, or refuses
 * the drafter naming the field.  Exit 0 when it opens, 2 when it refuses.
 * Usage: test_dflash_open MODEL [DRAFTER|-] [CONTEXT] [DRAFT_TOKENS] */
#include "../ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "Usage: %s MODEL [DRAFTER|-] [CONTEXT] [DRAFT_TOKENS]\n", argv[0]);
        return 1;
    }
    const char *drafter = argc > 2 && strcmp(argv[2], "-") != 0 ? argv[2] : NULL;
    ds4_engine_options opt = {
        .model_path = argv[1],
        .mtp_path = drafter,
        .context_size = argc > 3 ? atoi(argv[3]) : 32768,
        .mtp_draft_tokens = argc > 4 ? atoi(argv[4]) : 0,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        printf("open refused\n");
        return 2;
    }
    printf("open ok: qwen4=%d has_dflash=%d\n",
           ds4_engine_is_qwen4(engine), ds4_engine_has_dflash(engine));
    ds4_engine_close(engine);
    return 0;
}
