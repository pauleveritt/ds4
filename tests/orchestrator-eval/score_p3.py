import re, glob, os
def body(path):
    t=open(path,errors='ignore').read()
    for m in ["... (truncated)","Be terse and factual."]:
        i=t.rfind(m)
        if i!=-1: t=t[i+len(m):]; break
    return re.split(r"\[ Prompt: |\nExiting\.\.\.", t)[0].strip()
P3=[
 # --- must INFER these; the user-story spec never names them ---
 ("INFER models.py",      lambda s: "models.py" in s),
 ("INFER complaints.html",lambda s: "complaints.html" in s),
 ("INFER dataclass",      lambda s: re.search(r"dataclass", s, re.I) is not None),
 ("INFER timestamp field",lambda s: "timestamp" in s),
 ("INFER app.py route",   lambda s: "app.py" in s),
 # --- literals supplied verbatim in the spec ---
 ("lit Complaints Board", lambda s: "Complaints Board" in s),
 ("lit Scope creep",      lambda s: "Scope creep never ends" in s),
 ("lit agent_name/text",  lambda s: "agent_name" in s and re.search(r"`text`|\btext\b", s) is not None),
 # --- the behavioural trap: distinct instants => per-instance default ---
 ("TRAP default_factory", lambda s: re.search(r"default_factory|per-instance|evaluated (once|at )|import time|field\(", s, re.I) is not None),
 ("timezone-aware",       lambda s: re.search(r"timezone|tz-aware|utc", s, re.I) is not None),
 # --- packet contract ---
 ("marks NEW/SHARED",     lambda s: "NEW" in s and "SHARED" in s),
 ("preserve phase 1",     lambda s: re.search(r"__main__|uvicorn|home\.html|tagline|/ route|GET /\b", s) is not None),
 ("validation cmd",       lambda s: ".venv/bin/python -m pytest tests/" in s),
 ("no stdlib shadowing",  lambda s: not re.search(r"\b(dataclasses|types|json|datetime|typing|models)\.py\b", s) or "models.py" in s),
 ("app.py marked SHARED",  lambda s: re.search(r"SHARED[^\n]{0,80}app\.py|app\.py[^\n]{0,40}SHARED", s, re.S) is not None),
 ("no inlined code",       lambda s: not re.search(r"<h1|\{\{|\{%|strftime|@dataclass", s)),
 ("NO code fences",       lambda s: s.count("```")<=1),
 ("NO phase-3 leak",      lambda s: not re.search(r"RedirectResponse|303|POST /complaints|<form", s, re.I)),
]
files=sorted(glob.glob("/tmp/mellum-eval/out/p3-*.prompt_p3.txt"))
labels=[os.path.basename(f).split('.')[0] for f in files]
bodies=[body(f) for f in files]
print(f"{'check':<24}"+"".join(f"{l.replace('p3-',''):>16}" for l in labels))
for n,fn in P3:
    print(f"{n:<24}"+"".join(f"{('YES' if fn(b) else '--'):>16}" for b in bodies))
print(f"{'words':<24}"+"".join(f"{len(b.split()):>16}" for b in bodies))
print(f"{'SCORE':<24}"+"".join(f"{str(sum(1 for _,fn in P3 if fn(b)))+'/18':>16}" for b in bodies))
