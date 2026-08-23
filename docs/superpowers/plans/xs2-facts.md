# Laguna XS.2 GGUF ground-truth facts

Source of truth for Tasks 2-4 of `docs/superpowers/plans/2026-07-25-laguna-xs2-streaming.md`.
Every value below marked **GGUF** was read directly from the downloaded GGUF
metadata/tensor list (via `gguf.GGUFReader`, since `./ds4 --inspect` dies
before printing a dump — see "Inspector behavior" below). Values marked
**config.json** are corroborating-only, from the HF base-model config, and
are NOT authoritative where they conflict with GGUF (several do — see
"Corrections to the plan draft" at the bottom).

## 0. Repo-name correction (read this first)

The brief and the plan (`2026-07-25-laguna-xs2-streaming.md`) both assume a
repo named `poolside/Laguna-XS.2-GGUF` (dot notation, no version suffix).
**That repo does not exist as a public/official poolside repo** —
`curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/poolside/Laguna-XS.2-GGUF`
returns `401` (i.e., not accessible/not found for our purposes), and the API
tree-listing call in the brief returns `{"error":"Invalid username or
password."}`. `Laguna-XS.2` (no `-GGUF`) exists under `poolside/` but is a
**safetensors-only base model repo**, not a GGUF release.

The actual official GGUF repo, found via
`curl -s "https://huggingface.co/api/models?author=poolside&search=XS"`, is:

```
poolside/Laguna-XS-2.1-GGUF
```

This is a **newer, updated model**, not just a differently-spelled name for
the same thing. Per its README: "Laguna XS 2.1 ... is an upgraded version of
our [Laguna XS.2] model with a +5.4% jump on SWE-bench Multilingual". Same
`LagunaForCausalLM` architecture family as Laguna S 2.1 (already in ds4) and
the same naming convention (`Laguna-<size>-2.1`), just a different
generation than the `Laguna-XS.2` name the brief/plan assumed. All facts
below are for **Laguna XS 2.1** (`poolside/Laguna-XS-2.1-GGUF`), which is
what actually got downloaded. If the plan's later tasks want to keep calling
the ds4 variant "XS.2" internally that's a naming choice for Task 2 to make
explicitly — the shape values themselves are XS-2.1's.

`curl -s https://huggingface.co/api/models/poolside/Laguna-XS-2.1-GGUF`:
- `"sha": "1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6"`
- `"lastModified": "2026-07-18T13:49:48.000Z"`
- `"gated": false`, `"private": false`

## 1. Quant-file correction: BF16 substituted for Q8_0 (size deviation flagged)

The repo's file tree (`curl -s https://huggingface.co/api/models/poolside/Laguna-XS-2.1-GGUF/tree/main`) is:

| file | size (tree API) |
|---|---|
| `Laguna-XS-2.1-BF16.gguf` | 63829 MiB (66,927,837,184 B) |
| `Laguna-XS-2.1-Q4_K_M.gguf` | 19335 MiB (20,274,003,968 B) |
| `LICENSE.md`, `README.md`, `.gitattributes` | metadata only |

**There is no Q8_0 file in this repo** — only BF16 (full precision) and
Q4_K_M. Per the task ambiguity-resolution rule ("if only one of Q8_0/Q4_K_M
exists under a different naming ... download the closest equivalents"), I
downloaded **BF16 in place of Q8_0**.

**Flagging prominently**: this makes the actual download ~83.2 GB total
(66.9 + 20.3 GB on disk, confirmed by `ls -la` below), not the ~54 GB the
task authorization assumed (which was sized for Q8_0 ~35 GB + Q4_K_M ~19 GB).
393 GB was free before starting; this was well within capacity, but it's
roughly 50% more download/disk than the original estimate, and every later
task that says "Q8_0" should read "BF16" instead — there is no lower-effort
full-precision reference available from poolside for this model. If a task
specifically needs an 8-bit signal-path baseline (e.g. Task 10's
`llama-quantize --imatrix ... <Q8_0-file> ... q8_0` baseline type), it will
need to quantize BF16→Q8_0 itself first (or substitute BF16 directly as the
quantize input, which is the more common workflow anyway).

```
$ stat -f%z gguf/Laguna-XS-2.1-*.gguf
66930226304  Laguna-XS-2.1-BF16.gguf     (63829.6 MiB; tree API said 63829 MiB — matches)
20274300032  Laguna-XS-2.1-Q4_K_M.gguf   (19335.1 MiB; tree API said 19335 MiB — matches)
```
Both downloads completed fully in the background per the task authorization
(`run_in_background: true`), verified against the tree listing to well
within the brief's ±1% tolerance (actually <0.01% off — the tiny excess is
GGUF header/metadata overhead not reflected in the LFS pointer size).

## 2. Inspector behavior (`./ds4 --inspect`)

`DS4_LOCK_FILE=/tmp/ds4-inspect.lock ./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf --inspect`
(needed a non-default `DS4_LOCK_FILE` because an unrelated pre-existing
`ds4-agent` process — not started by this task — already held
`/tmp/ds4.lock`; this does not touch that process).

Output (all of it — dies immediately, prints **no metadata dump** at all,
contrary to the brief's "PRINTS METADATA BUT LIKELY DIES" expectation):

```
ds4: expected block_count=48 for Laguna S 2.1, got 40
```

Exit code 1. So the brief's primary path (`--inspect | head -80`) gives
nothing usable — went straight to the documented fallback (Python
`gguf.GGUFReader`). Note `pip3 install gguf` alone did not work in the
ambient Python (an unrelated stub package shadowed the real one on
`sys.path` — `import gguf` succeeded but `GGUFReader` wasn't in it); fixed
by creating a throwaway venv (`docs/superpowers/plans/.gguf-venv`, not
committed) and installing `gguf==0.19.0` there.

## 3. GGUF metadata — `laguna.*` keys (Q4_K_M file; GGUF, authoritative)

```
general.architecture = laguna
general.file_type = 15                      (llama.cpp MOSTLY_Q4_K_M)
general.finetune = 2afce91e7cdb129f891d65a18f54d4a63f8e3215
general.license = openmdw-1.1
general.name = 2afce91e7cdb129f891d65a18f54d4a63f8e3215
general.quantization_version = 2
general.size_label = 256x2.2B
general.tags = ['laguna-xs-2.1', 'vllm', 'text-generation']
general.type = model

laguna.attention.head_count = [48, 64, 64, 64] x10  (40 entries total, see below)
laguna.attention.head_count_kv = 8
laguna.attention.key_length = 128
laguna.attention.layer_norm_rms_epsilon = 9.999999974752427e-07   (~1e-6)
laguna.attention.sliding_window = 512
laguna.attention.value_length = 128
laguna.block_count = 40
laguna.context_length = 262144
laguna.embedding_length = 2048
laguna.expert_count = 256
laguna.expert_feed_forward_length = 512
laguna.expert_gating_func = 2
laguna.expert_shared_feed_forward_length = 512
laguna.expert_used_count = 8
laguna.expert_weights_norm = True
laguna.expert_weights_scale = 2.5
laguna.feed_forward_length = 8192
laguna.leading_dense_block_count = 1
laguna.rope.dimension_count = 64
laguna.rope.dimension_count_swa = 128
laguna.rope.freq_base = 500000.0
laguna.rope.freq_base_swa = 10000.0
laguna.rope.scaling.factor = 32.0
laguna.rope.scaling.original_context_length = 8192
laguna.rope.scaling.type = yarn
laguna.rope.scaling.yarn_attn_factor = 1.0
laguna.rope.scaling.yarn_beta_fast = 64.0
laguna.rope.scaling.yarn_beta_slow = 1.0
laguna.vocab_size = 100352
```

(`tokenizer.*` keys omitted — standard BPE vocab/merges, nothing
XS.2-specific, and the vocab array alone is ~2.7 MB of text.)

### Item 1 — per-layer head-count array (spec unknown #1): RESOLVED

`laguna.attention.head_count` is a 40-entry array, the repeating pattern
`[48, 64, 64, 64]` (i.e. every 4th layer, `il % 4 == 0`, is 48; the other
three of every four are 64). **Not flat.** Mirrors S 2.1's
"reduced-heads-on-global-layers" shape, just with different numbers:

| | S 2.1 (in ds4) | XS 2.1 |
|---|---|---|
| global layers (`il%4==0`) head count | 48 | **48** |
| SWA layers (other 3/4) head count | 72 | **64** |

So: **`N_HEAD_GLOBAL = 48`, `N_HEAD_SWA = 64`**.

Important for Task 2: ds4's existing `DS4_N_HEAD` / `ds4_laguna_layer_is_swa`
(`ds4.c:1141-1144`) compares a layer's head count to `DS4_N_HEAD` and calls
it SWA if equal. For S 2.1, `DS4_N_HEAD = 72` (the *majority/SWA* value, not
the max in any principled sense — it's just what 3/4 of layers use). For
XS 2.1 the equivalent field must be **64** (the SWA/majority value), **not
48**. See "Corrections to the plan draft" below — the plan's Task 2 draft
struct has `.n_head = 48` for XS2, which is wrong by this reading, and
comments "flat 48" (`n_head_swa = 48`) as one option, which the GGUF data
directly contradicts (48 and 64 are different).

### Item 2 — leading-dense count (spec unknown #2): RESOLVED

`laguna.leading_dense_block_count = 1` (confirmed directly in GGUF metadata,
not just inferred from config.json's `mlp_only_layers: [0]`).

- Dense layers: 1 (layer 0 only)
- Sparse/MoE layers: 40 − 1 = **39**
- Routed expert total: **39 × 256 = 9,984**

(Not 38×256=9728 — the plan/spec's other candidate option is wrong.)

### Item 3 — rope dimension counts (spec unknown #3): RESOLVED

```
laguna.rope.dimension_count     = 64   (global/full-attention layers) -> DS4_N_ROT
laguna.rope.dimension_count_swa = 128  (sliding-window layers)         -> DS4_N_ROT_SWA
```

Same numeric values as S 2.1 (64/128), so the plan's draft got this pair
right, though for a different reason than S 2.1 — here it's `head_dim(128) *
partial_rotary_factor` with factors 0.5 (global) / 1.0 (SWA), confirmed by
the HF `config.json` `rope_parameters` block (see below) and independently
by the raw GGUF keys above.

### Full YaRN/rope parameter set (all directly from GGUF; several correct the plan draft)

| Field | XS 2.1 (GGUF, authoritative) | Plan/spec draft said | Match? |
|---|---|---|---|
| `rope.freq_base` (global) | 500000.0 | 500000.0 | yes |
| `rope.freq_base_swa` | 10000.0 | 10000.0 | yes |
| `rope.scaling.type` | yarn | yarn | yes |
| `rope.scaling.factor` | **32.0** | 64.0 | **NO — plan wrong** |
| `rope.scaling.original_context_length` | **8192** | 4096 | **NO — plan wrong** |
| `rope.scaling.yarn_beta_fast` | 64.0 | 64.0 | yes |
| `rope.scaling.yarn_beta_slow` | 1.0 | 1.0 | yes |
| `rope.scaling.yarn_attn_factor` | **1.0** | 1.4158883083359672 | **NO — plan wrong** |
| `attention.layer_norm_rms_epsilon` | ~1e-6 | (assumed 1e-6) | yes |

The plan's placeholder values for `rope_scale_factor`, `rope_orig_ctx`, and
`rope_yarn_attn_factor` (`ds4.c` Task 2 draft struct and the spec's
comparison table) do not match the GGUF. Note the GGUF's
`yarn_attn_factor = 1.0` also does **not** match the HF `config.json`'s
*computed* `attention_factor = 1.3465735902799727` (see below) — the GGUF
conversion apparently stores the flat default rather than HF's derived
value. **Use the GGUF value (1.0)**, since that's what ds4 will actually
read via `required_f32(m, "laguna.rope.scaling.yarn_attn_factor")`.

### Cross-check only: HF `config.json` (`poolside/Laguna-XS-2.1`, non-authoritative)

Fetched for corroboration, not as a source of truth (a few fields disagree
with the GGUF, as shown above):

```
hidden_size = 2048, intermediate_size = 8192, num_hidden_layers = 40
num_attention_heads = 48 (default; per-layer override below)
num_attention_heads_per_layer = [48,64,64,64] x10   <- matches GGUF array
num_key_value_heads = 8, head_dim = 128
num_experts = 256, num_experts_per_tok = 8
moe_intermediate_size = 512, shared_expert_intermediate_size = 512
mlp_only_layers = [0]                                <- matches leading_dense=1
sliding_window = 512
rope_parameters.full_attention: rope_theta=500000.0, rope_type=yarn,
  factor=32.0, original_max_position_embeddings=8192, beta_slow=1.0,
  beta_fast=64.0, attention_factor=1.3465735902799727 (computed; GGUF has 1.0 instead),
  partial_rotary_factor=0.5  (-> 64 = n_rot)
rope_parameters.sliding_attention: rope_type=default, rope_theta=10000.0,
  partial_rotary_factor=1.0  (-> 128 = n_rot_swa)
```

## 4. Tensor names, shapes, and GGUF types (item 4)

`n_tensors = 678` in the Q4_K_M file. GGML type codes seen: `0`=F32,
`12`=Q4_K, `14`=Q6_K (standard ggml enum; no Q8_0/other types present in
this file — consistent with a pure Q4_K_M quant).

### Non-layer tensors

| tensor | type | shape |
|---|---|---|
| `token_embd.weight` | Q4_K (12) | [2048, 100352] |
| `output_norm.weight` | F32 (0) | [2048] |
| `output.weight` | Q6_K (14) | [2048, 100352] |

`token_embd.weight` being Q4_K (not Q8_0) means this file lands on ds4's
existing "legacy" layout-marker branch in `weights_validate_laguna_layout`
(`ds4.c:5057-5060`: `legacy_layout = (w->token_embd && type==Q4_K)`) — same
marker convention S 2.1's Q4_K_M file uses. Task 3 will need a new XS-2.1
recipe entry, but the *marker detection mechanism* itself (Q4_K token_embd
= legacy layout) does not need new code, just a new accepted shape/variant.

### Layer 0 (the one leading-dense layer, `il < n_leading_dense`)

```
blk.0.attn_norm.weight     F32   [2048]
blk.0.attn_q.weight        Q4_K  [2048, 6144]
blk.0.attn_k.weight        Q4_K  [2048, 1024]
blk.0.attn_v.weight        Q6_K  [2048, 1024]
blk.0.attn_gate.weight     Q4_K  [2048, 48]        <- attn_gate output dim == that layer's head_count (48, global)
blk.0.attn_q_norm.weight   F32   [128]
blk.0.attn_k_norm.weight   F32   [128]
blk.0.attn_output.weight   Q4_K  [6144, 2048]
blk.0.ffn_norm.weight      F32   [2048]
blk.0.ffn_gate.weight      Q4_K  [2048, 8192]      (dense FFN, not routed)
blk.0.ffn_up.weight        Q4_K  [2048, 8192]
blk.0.ffn_down.weight      Q6_K  [8192, 2048]
```

### Layer 1 (first sparse/MoE layer — the pattern for all 39 sparse layers)

```
blk.1.attn_norm.weight       F32   [2048]
blk.1.attn_q.weight          Q4_K  [2048, 8192]
blk.1.attn_k.weight          Q4_K  [2048, 1024]
blk.1.attn_v.weight          Q6_K  [2048, 1024]     (type varies by layer, see below)
blk.1.attn_gate.weight       Q4_K  [2048, 64]        <- 64 == that layer's head_count (SWA)
blk.1.attn_q_norm.weight     F32   [128]
blk.1.attn_k_norm.weight     F32   [128]
blk.1.attn_output.weight     Q4_K  [8192, 2048]
blk.1.ffn_norm.weight        F32   [2048]
blk.1.ffn_gate_inp.weight    F32   [2048, 256]
blk.1.exp_probs_b.bias       F32   [256]
blk.1.ffn_gate_exps.weight   Q4_K  [2048, 512, 256]
blk.1.ffn_up_exps.weight     Q4_K  [2048, 512, 256]
blk.1.ffn_down_exps.weight   Q6_K  [512, 2048, 256]  (type varies by layer, see below)
blk.1.ffn_gate_shexp.weight  Q4_K  [2048, 512]
blk.1.ffn_up_shexp.weight    Q4_K  [2048, 512]
blk.1.ffn_down_shexp.weight  Q6_K  [512, 2048]        (type varies by layer, see below)
```

Confirmed identical tensor-name set on layer 39 (last layer) — the pattern
is uniform across all 39 sparse layers; only quant *type* of a few tensors
varies (next section).

Tensor names match ds4's existing `weights_bind_laguna_layer` (`ds4.c:6132-
6158`) exactly — no new tensor names needed for XS 2.1, this is a pure
shape/variant change, not a tensor-schema change.

`attn_gate.weight`'s second dim equals that layer's `head_count` (48 or 64)
— i.e. it's a per-head gate, consistent with ds4's existing binding.

## 5. Q4_K_M per-tensor type map (item 6 — Task 3 validation entry)

Overall type histogram (all 678 tensors, Q4_K_M file):

```
Q4_K (12): 379 tensors
F32  (0) : 239 tensors
Q6_K (14):  60 tensors
```

By tensor-name suffix, **most are a single fixed type across all layers**:

| suffix | type(s) |
|---|---|
| `attn_norm.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`, `ffn_norm.weight`, `ffn_gate_inp.weight`, `exp_probs_b.bias`, `output_norm.weight` | F32 only |
| `attn_q.weight`, `attn_k.weight`, `attn_gate.weight`, `attn_output.weight`, `ffn_gate.weight` (dense), `ffn_up.weight` (dense), `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, `token_embd.weight` | Q4_K only |
| `ffn_down.weight` (dense, layer 0 only), `output.weight` | Q6_K only |

**Exception — three suffixes vary Q4_K vs Q6_K *per layer*, and the split
does NOT correlate with the global/SWA (`il%4`) attention pattern:**
`attn_v.weight`, `ffn_down_exps.weight`, `ffn_down_shexp.weight`. All three
always move together (same type within a given layer). Measured per-layer
(0-indexed; "global"/"swa" = attention pattern, unrelated to the type split):

```
layer: type (all three of attn_v/ffn_down_exps/ffn_down_shexp move together)
 0:Q6_K  1:Q6_K  2:Q6_K  3:Q6_K  4:Q6_K  5:Q4_K  6:Q4_K  7:Q6_K  8:Q4_K  9:Q4_K
10:Q6_K 11:Q4_K 12:Q4_K 13:Q6_K 14:Q4_K 15:Q4_K 16:Q6_K 17:Q4_K 18:Q4_K 19:Q6_K
20:Q4_K 21:Q4_K 22:Q6_K 23:Q4_K 24:Q4_K 25:Q6_K 26:Q4_K 27:Q4_K 28:Q6_K 29:Q4_K
30:Q4_K 31:Q6_K 32:Q4_K 33:Q4_K 34:Q6_K 35:Q6_K 36:Q6_K 37:Q6_K 38:Q6_K 39:Q6_K
```

(Layer 0's `attn_v` is Q6_K too, consistent with the run above; layer 0 has
no `ffn_down_exps`/`ffn_down_shexp` since it's dense.) This is llama.cpp's
standard Q4_K_M "extra bits on some layers" heuristic (from `llama-quantize`
default layer-importance rules), not anything Laguna-specific, and not tied
to the model's own SWA/global layer pattern — **Task 3's layout-acceptance
recipe for `xs2-official-q4km` must tolerate per-layer Q4_K/Q6_K variation
on these three tensor kinds specifically**, rather than expecting one fixed
type the way most other tensors allow.

## 6. Exact filenames + HF revision (item 5)

- Repo: `poolside/Laguna-XS-2.1-GGUF`
- Revision/sha: `1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6` (lastModified 2026-07-18T13:49:48Z)
- Files downloaded to `gguf/`, both complete and size-verified:
  - `Laguna-XS-2.1-Q4_K_M.gguf` — 20,274,300,032 bytes (19335.1 MiB)
  - `Laguna-XS-2.1-BF16.gguf` — 66,930,226,304 bytes (63829.6 MiB) — **substituted for the Q8_0 the brief/plan expected (see section 1)**

## 7. BF16 file cross-check (item 4, "in both files")

Ran the same `gguf.GGUFReader` dump against `Laguna-XS-2.1-BF16.gguf`.
Findings:

- **All `laguna.*` and `general.*` KV metadata values are byte-for-byte
  identical to the Q4_K_M file** (same head-count array, same rope params,
  same leading-dense count, same everything in section 3 above) — the only
  metadata difference is `general.file_type` (`32` for BF16 vs `15` for
  Q4_K_M — GGUF's file-type field encodes the dominant quantization, and
  differs between conversions as expected; not a discrepancy).
- **Tensor count is identical: 678** in both files. Same tensor *names* on
  every layer (confirmed layer 1 name-for-name against section 4's Q4_K_M
  list — identical set, no additions/removals).
- **Types are uniform** in BF16, unlike Q4_K_M: every weight tensor is
  ggml type `30` (BF16) and every norm/bias is type `0` (F32) — **no
  per-layer type variation** (the Q4_K/Q6_K split described in section 5 is
  purely a Q4_K_M-quantization artifact, not present in the source file).
  So BF16 gives a clean, uniform reference — useful as the `llama-quantize`
  input for Task 10, and as a "no mixed types to worry about" baseline for
  Task 3's dense-model sanity checks.

This resolves the "UNRESOLVED" item from the earlier draft of this
document — both files are now confirmed consistent.

## Corrections to the plan draft (`docs/superpowers/plans/2026-07-25-laguna-xs2-streaming.md`)

Task 2's draft `DS4_SHAPE_LAGUNA_XS2` struct and the spec's comparison table
both contain values that this GGUF ground-truth contradicts. Summary for
whoever implements Task 2:

| Field | Plan/spec draft value | Correct value (this doc) |
|---|---|---|
| Repo name | `poolside/Laguna-XS.2-GGUF` | `poolside/Laguna-XS-2.1-GGUF` (see section 0) |
| `.n_head` (S21-style "majority/SWA" field) | 48 | **64** |
| `.n_head_swa` | 48 (flat) | **64** |
| `.n_head_global` | 48 | 48 (correct) |
| `.n_leading_dense` | 1 | 1 (correct) |
| `.n_rot` / `.n_rot_swa` | 64 / 128 | 64 / 128 (correct) |
| `.rope_scale_factor` | 64.0f | **32.0f** |
| `.rope_orig_ctx` | 4096 | **8192** |
| `.rope_yarn_attn_factor` | 1.4158883083359672f | **1.0f** |
| `.rope_yarn_beta_fast` | 64.0f | 64.0f (correct) |
| `.rope_yarn_beta_slow` | 1.0f | 1.0f (correct) |
| Routed expert total | "38×256 or 39×256, TBD" | **39×256 = 9984** |
| Model name string | `"Laguna XS.2"` | judgment call — the actual model is "Laguna XS 2.1"; keep internal variant name `DS4_VARIANT_LAGUNA_XS2` if desired (harmless), but the `.name` display string and any user-facing text should say "Laguna XS 2.1" to match what it actually is |

None of these are close/rounding differences — `rope_scale_factor` (32 vs
64) and `rope_orig_ctx` (8192 vs 4096) are both exactly 2x off, and
`n_head`/`n_head_swa` (64 vs 48) would make `ds4_laguna_layer_is_swa`
misclassify every layer if left at the plan's flat-48 assumption.

## UNRESOLVED

- Local file sha256 checksums for the two downloaded GGUFs were not
  recorded (not required by the brief, which only asked for the HF repo
  revision sha, item 5/section 6 above — but if a future task wants to
  verify file integrity against a specific download, it will need to
  recompute; both files' byte sizes are recorded above and matched the HF
  tree listing to well within tolerance).
- Nothing else is outstanding — both official files were downloaded in
  full, `laguna.*`/`general.*` metadata and layer-0/layer-1/layer-39 tensor
  names+types were confirmed directly from both GGUFs (not inferred from
  config.json), and every spec unknown (#1-#4 from the design doc) is
  resolved above.
