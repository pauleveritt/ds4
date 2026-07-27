### Task 4: Quality fixtures for XS 2.1

**Files:**
- Create: `gguf-tools/quality-testing/data/laguna-xs21/` (continuations)
- Modify: `gguf-tools/quality-testing/collect_official.py` (model entry)

**Interfaces:**
- Consumes: working resident XS 2.1 (Task 3).
- Produces: fixture continuations used as the quality baseline in Tasks 10 and 13. Reference source: OpenRouter XS 2.1 if listed (spec unknown #5); otherwise laptop BF16 output is the reference — record which in the fixture README.

- [ ] **Step 1: Check OpenRouter availability**

```bash
curl -s https://openrouter.ai/api/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data'] if 'laguna' in m['id'].lower()])"
```

- [ ] **Step 2: Collect ~30 fixture cases** following the existing S 2.1 fixture pattern in `gguf-tools/quality-testing/` (the S 2.1 commit added `continuations/case_*.txt` + README). Prompt mix: 15 web/Python (HTML/CSS/JS/TS/Python), 10 general prose/reasoning, 5 tool-call format. Use `collect_official.py` with the OpenRouter id if available, else run the BF16 locally with the same prompts.

- [ ] **Step 3: Verify the local Q4_K_M against fixtures** using the existing quality-testing procedure (its README documents the comparison run). Record the agreement score in the fixture README as the P0 baseline.

- [ ] **Step 4: Commit**

```bash
git add gguf-tools/quality-testing/
git commit -m "Add Laguna XS 2.1 quality fixtures and P0 baseline"
```

---

## Phase P1 — Streaming port

