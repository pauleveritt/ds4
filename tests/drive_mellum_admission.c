#include "ds4.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CPU;
    opt.inspect_only = true;
    opt.load_slice = true;
    opt.load_layer_start = 1;
    opt.load_layer_end = 1;
    opt.load_output = false;

    ds4_engine *e = NULL;
    int rc = ds4_engine_open(&e, &opt);
    if (rc != 0) {
        fprintf(stderr, "ds4_engine_open failed (rc=%d)\n", rc);
        return 1;
    }
    ds4_engine_close(e);
    printf("ADMISSION_OK\n");
    return 0;
}
