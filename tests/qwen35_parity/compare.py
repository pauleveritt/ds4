"""Compare two parity runs (ds4_side.sh or llama_side.py output directories).

usage: compare.py A_DIR B_DIR [--table]

Per prompt: whether the greedy tokens agree and where they first diverge
(with both sides' top-2 at that step), whether the top-5 ids at the last
prompt position agree in order, and the absolute logprob difference of the
chosen token over the agreeing steps.  --table prints the step-0 top-5 of
both sides as a markdown table instead.
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    j = json.load(open(path))
    if "steps" in j:
        steps = [{"id": s["selected"]["id"], "top": [(t["token"]["id"], t["logprob"]) for t in s["top_logprobs"]]}
                 for s in j["steps"]]
        return j["prompt_tokens"], steps
    probs = j["completion"]["completion_probabilities"]
    steps = [{"id": c["id"], "top": [(t["id"], t["logprob"]) for t in c["top_logprobs"]]} for c in probs]
    return len(j["ds4_tokens"]), steps


def summary(name, a, b):
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i]["id"] != b[i]["id"]), None)
    upto = n if div is None else div + 1
    diffs = []
    for i in range(upto):
        other = dict(b[i]["top"])
        tid, lp = a[i]["top"][0]
        if tid in other:
            diffs.append(abs(lp - other[tid]))
    same_top = [t for t, _ in a[0]["top"]] == [t for t, _ in b[0]["top"]]
    if div is None:
        tokens = f"all {n} tokens match"
    else:
        ta, tb = a[div]["top"], b[div]["top"]
        tokens = (f"diverge at step {div} (A {ta[0][0]}:{ta[0][1]:.3f} {ta[1][0]}:{ta[1][1]:.3f}; "
                  f"B {tb[0][0]}:{tb[0][1]:.3f} {tb[1][0]}:{tb[1][1]:.3f})")
    print(f"{name:11s} {tokens}; step-0 top-5 ids {'equal' if same_top else 'DIFFER'}; "
          f"chosen-token |dlogprob| median {statistics.median(diffs):.4f} max {max(diffs):.4f}")


def table_row(name, n_prompt, a, b):
    ta, tb = a[0]["top"], b[0]["top"]
    same = [t for t, _ in ta] == [t for t, _ in tb]
    worst = max(abs(x[1] - y[1]) for x, y in zip(ta, tb)) if same else float("nan")
    fa = ", ".join(f"{i}: {v:.3f}" for i, v in ta)
    fb = ", ".join(f"{i}: {v:.3f}" for i, v in tb)
    print(f"| {name} | {n_prompt} | {fa} | {fb} | {'yes' if same else 'no'} | {worst:.3f} |")


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    table = "--table" in sys.argv[3:]
    if table:
        print("| Prompt | Tokens | A top-5 (id: logprob) | B top-5 (id: logprob) | Same ids, same order | Max abs. diff |")
        print("|---|---|---|---|---|---|")
    for p in json.load(open(os.path.join(HERE, "prompts.json"))):
        na, a = load(os.path.join(a_dir, p["name"] + ".json"))
        nb, b = load(os.path.join(b_dir, p["name"] + ".json"))
        if na != nb:
            print(f"{p['name']}: prompt lengths differ ({na} vs {nb})")
        elif table:
            table_row(p["name"], na, a, b)
        else:
            summary(p["name"], a, b)


if __name__ == "__main__":
    main()
