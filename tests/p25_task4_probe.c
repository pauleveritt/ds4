/* P25 Task 4: mechanism demo, not a speed claim.
 *
 * Proves, via the real product session API (ds4_session_create /
 * ds4_session_sync -- the same functions ds4-agent's product path calls,
 * not a probe-only code path), that the Mellum layer-major sync's
 * prefill-scratch cap freezes at whatever the FIRST sync >= 64 pending
 * tokens sees (min(pending, DS4_N_SWA)), and stays frozen for the rest of
 * the session's life -- so a session whose first sync is small locks in
 * small chunks for every later, larger span, even though DS4_N_SWA=1024
 * would have allowed much bigger chunks.
 *
 * Chunk count is read directly from the progress callback
 * ("prefill_chunk" events), not from timing -- a mechanism proof, per the
 * plan's Task 4 scope decision, not a speed claim.
 *
 * Standalone, deliberately outside tests/ds4_test.c's shared harness: it
 * needs full control over exact prompt token counts (ds4-agent's fixed
 * ~2,127-token prompt-prefix always saturates the cap at DS4_N_SWA on the
 * very first sync, which is itself a finding recorded in the close-out, but
 * makes ds4-agent unusable for demonstrating the SMALL-first-sync case).
 *
 * Build: cc -O2 -std=c99 -I.. -o p25_task4_probe p25_task4_probe.c \
 *   ../ds4.o ../ds4_image.o ../ds4_distributed.o ../ds4_tp.o ../ds4_ssd.o \
 *   ../ds4_metal.o ../ds4_layer_pack.o ../ds4_engram.o -lm -pthread \
 *   -framework Foundation -framework Metal
 * Run: DS4_TEST_MODEL=<Q8_0 path> ./p25_task4_probe
 */
#include "../ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_chunk_events;

static void progress_cb(void *ud, const char *event, int current, int total) {
    (void)ud;
    (void)current;
    (void)total;
    if (event && strcmp(event, "prefill_chunk") == 0) g_chunk_events++;
}

static void build_tokens(ds4_tokens *t, int n, int start_id) {
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < n; i++) {
        ds4_tokens_push(t, start_id + (i % 50) + 3);
    }
}

int main(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    if (!model_path || !model_path[0]) {
        fprintf(stderr, "p25_task4_probe: set DS4_TEST_MODEL\n");
        return 2;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_METAL;

    /* ds4_engine_open_for_agent, not ds4_engine_open: Mellum resident
     * sessions are only enabled for the agent-owner caller
     * (e->mellum_interactive_sessions = agent_owner && !inspect_only,
     * ds4.c:75279) -- this probe needs the same real product session path
     * ds4-agent uses, not the probe-only inspect_only path. */
    ds4_engine *e = NULL;
    if (ds4_engine_open_for_agent(&e, &opt) != 0) {
        fprintf(stderr, "p25_task4_probe: ds4_engine_open_for_agent failed\n");
        return 1;
    }

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 4096) != 0) {
        fprintf(stderr, "p25_task4_probe: ds4_session_create failed\n");
        return 1;
    }
    ds4_session_set_progress(s, progress_cb, NULL);

    char err[256];

    /* First sync: 86 pending tokens (>= sync_batch_min_tokens=64), so the
     * product path takes the layer-major batched sync and freezes
     * s->mellum->prefill's cap at min(86, DS4_N_SWA) = 86. */
    ds4_tokens small;
    build_tokens(&small, 86, 0);
    g_chunk_events = 0;
    int rc1 = ds4_session_sync(s, &small, err, sizeof(err));
    int chunks_small = g_chunk_events;
    printf("P25-TASK4 first_sync pending=86 rc=%d chunk_events=%d\n",
           rc1, chunks_small);
    if (rc1 != 0) {
        fprintf(stderr, "p25_task4_probe: first sync failed: %s\n", err);
        return 1;
    }

    /* Second sync: extends the checkpoint by 2,000 NEW tokens (total 2,086).
     * Common-prefix reuse means only the new 2,000 are "pending". Since the
     * cap froze at 86 on the first sync, this must chunk in ~86-token
     * pieces (~24 chunks) rather than the ~2 chunks a cap of 1,024 would
     * give -- exactly the cost Task 4 asks to demonstrate. */
    ds4_tokens big;
    build_tokens(&big, 86 + 2000, 1);
    /* Keep the first 86 identical to `small` so the session's common-prefix
     * check treats them as the same prefix (ds4_tokens_starts_with). */
    for (int i = 0; i < 86; i++) big.v[i] = small.v[i];
    g_chunk_events = 0;
    int rc2 = ds4_session_sync(s, &big, err, sizeof(err));
    int chunks_big = g_chunk_events;
    printf("P25-TASK4 second_sync pending=2000 rc=%d chunk_events=%d "
           "expected_if_cap_1024=%d expected_if_cap_frozen_at_86=%d\n",
           rc2, chunks_big, (2000 + 1023) / 1024, (2000 + 85) / 86);
    if (rc2 != 0) {
        fprintf(stderr, "p25_task4_probe: second sync failed: %s\n", err);
        return 1;
    }

    ds4_tokens_free(&small);
    ds4_tokens_free(&big);
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
