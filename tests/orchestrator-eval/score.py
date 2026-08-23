import re, glob, os
def body(path):
    t = open(path, errors='ignore').read()
    # generation begins after the echoed (possibly truncated) prompt
    for marker in ["... (truncated)", "Be terse and factual.", "say so explicitly."]:
        i = t.rfind(marker)
        if i != -1:
            t = t[i+len(marker):]; break
    t = re.split(r"\[ Prompt: |\nExiting\.\.\.", t)[0]
    return t.strip()

P2 = [
 ("marks NEW",            lambda s: bool(re.search(r"\bNEW\b", s))),
 ("marks SHARED",         lambda s: bool(re.search(r"\bSHARED\b", s))),
 ("models.py",            lambda s: "models.py" in s),
 ("complaints.html",      lambda s: "complaints.html" in s),
 ("app.py SHARED",        lambda s: "app.py" in s),
 ("test_app.py SHARED",   lambda s: "test_app.py" in s),
 ("PRESERVE / route",     lambda s: bool(re.search(r"(GET )?[`\"]?/[`\"]?\s*(route|home|preserv)|home\.html|__main__|uvicorn", s, re.I))),
 ("lit: Complaints Board",lambda s: "Complaints Board" in s),
 ("lit: AgentClinic",     lambda s: "AgentClinic" in s),
 ("lit: tagline",         lambda s: "Tell us about your human" in s),
 ("lit: favicon URL",     lambda s: "python.org/static/favicon.ico" in s),
 ("exact validation cmd", lambda s: ".venv/bin/python -m pytest tests/" in s),
 ("semantic trap noted",  lambda s: bool(re.search(r"default_factory|evaluated once|import time|mutable default|timezone|isoformat|follow_redirects", s, re.I))),
 ("NO code fences",       lambda s: s.count("```") <= 1),
 ("NO phase-3 leak",      lambda s: not re.search(r"RedirectResponse|status.?303|POST /complaints|<form", s, re.I)),
]
P1 = [
 ("app.py",               lambda s: "app.py" in s),
 ("base.html",            lambda s: "base.html" in s),
 ("home.html",            lambda s: "home.html" in s),
 ("tests/test_app.py",    lambda s: "test_app.py" in s),
 ("lit: tagline",         lambda s: "Tell us about your human" in s),
 ("lit: favicon URL",     lambda s: "python.org/static/favicon.ico" in s),
 ("lit: AgentClinic",     lambda s: "AgentClinic" in s),
 ("starlette.testclient", lambda s: "starlette.testclient" in s),
 ("uvicorn.run app:app",  lambda s: 'uvicorn.run("app:app"' in s or "app:app" in s),
 ("validation cmd",       lambda s: "pytest" in s),
 ("NOTICED main.py clash",lambda s: bool(re.search(r"contradict|conflict|inconsisten|main\.py.*app\.py|app\.py.*main\.py", s, re.I))),
 ("NO code fences",       lambda s: s.count("```") <= 1),
 ("NO phase-2/3 leak",    lambda s: not re.search(r"Complaint\b|complaints\.html|dataclass", s)),
]
for phase, checks in (("prompt_p1", P1), ("prompt_p2", P2)):
    files = sorted(glob.glob(f"/tmp/mellum-eval/out/*.{phase}.txt"))
    if not files: continue
    labels = [os.path.basename(f).split('.')[0] for f in files]
    print(f"\n=== {phase} ===")
    print(f"{'check':<24}" + "".join(f"{l:>10}" for l in labels))
    bodies = [body(f) for f in files]
    for name, fn in checks:
        row = "".join(f"{('YES' if fn(b) else '--'):>10}" for b in bodies)
        print(f"{name:<24}{row}")
    print(f"{'words':<24}" + "".join(f"{len(b.split()):>10}" for b in bodies))
    print(f"{'SCORE':<24}" + "".join(f"{sum(1 for _,fn in checks if fn(b)):>7}/{len(checks):<2}" for b in bodies))
