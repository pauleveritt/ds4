"""Bounded RLM-readiness probe: does the model explore a large variable
programmatically, write valid code, and TERMINATE?"""
import json, re, sys, time, urllib.request, io, contextlib

DOC_PATH = "/Users/pauleveritt/projects/pauleveritt/local-ai-gemma/LESSONS.md"
QUERY = ("List every distinct mechanism this document recommends for bounding a "
         "runaway or looping implementer agent. For each, give the mechanism and "
         "the lesson number it appears in.")
MAX_HOPS = 6

SYSTEM = """You explore a large document that is NOT in your context. It is stored
in a Python variable named `doc` (a str) in a sandboxed REPL.

Each turn, reply with EITHER:
  1. A python code block to run, e.g.
     ```python
     print(len(doc))
     print([l for l in doc.split(chr(10)) if 'cap' in l][:5])
     ```
  2. Your final answer, prefixed exactly with FINAL:

Only `re`, `json`, `collections`, `itertools`, `textwrap` may be imported.
Use print() to see results. Keep each code block small. Stop as soon as you can
answer; do not keep exploring once you have enough."""

SAFE = {"re","json","collections","itertools","textwrap","math","string"}

def run_code(code, doc):
    if re.search(r"\b(os|sys|subprocess|shutil|socket|open|eval|exec|__import__)\b", code):
        return "ERROR: forbidden name"
    for m in re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]+)", code, re.M):
        if m not in SAFE: return f"ERROR: import {m} not allowed"
    buf = io.StringIO()
    ns = {"doc": doc, "print": print, "__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<rlm>", "exec"), ns)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    out = buf.getvalue()
    return out[:1800] + ("\n...[truncated]" if len(out) > 1800 else "") or "(no output)"

def chat(model, msgs, think, maxtok=2048):
    p = {"model": model, "messages": msgs, "max_tokens": maxtok,
         "temperature": 0.2, "top_p": 0.9}
    if think == "off":
        p["reasoning_effort"] = "none"
        p["chat_template_kwargs"] = {"enable_thinking": False}
    r = urllib.request.Request("http://127.0.0.1:8125/v1/chat/completions",
        data=json.dumps(p).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer evalkey"})
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    c = d["choices"][0]
    txt = c["message"].get("content") or ""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    return txt.strip(), d.get("usage",{}).get("completion_tokens",0), c.get("finish_reason")

def main(model, tag, think):
    doc = open(DOC_PATH).read()
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":f"`doc` holds a {len(doc)}-char document.\n\nQUERY: {QUERY}"}]
    tot, hops, valid, t0 = 0, 0, 0, time.time()
    final = None
    for hop in range(MAX_HOPS):
        txt, ntok, fin = chat(model, msgs, think)
        tot += ntok; hops += 1
        if "FINAL:" in txt:
            final = txt.split("FINAL:",1)[1].strip(); break
        m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
        if not m:
            msgs += [{"role":"assistant","content":txt},
                     {"role":"user","content":"Reply with a ```python block or FINAL: <answer>."}]
            continue
        valid += 1
        out = run_code(m.group(1), doc)
        msgs += [{"role":"assistant","content":txt},
                 {"role":"user","content":f"REPL output:\n{out}"}]
    el = time.time()-t0
    open(f"/tmp/mellum-eval/rlm/{tag}.txt","w").write(final or "(no final answer)")
    print(f"{tag:<18} hops={hops} code_blocks={valid} terminated={'YES' if final else 'NO':<3} "
          f"tokens={tot:<6} {el:6.1f}s")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
