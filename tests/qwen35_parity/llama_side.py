"""Reference side of the qwen35 parity check: llama.cpp's llama-server.

usage: llama_side.py MODEL TOKDIR OUTDIR [N_PREDICT]

For every prompt in prompts.json, sends the token ids ds4 rendered
(TOKDIR/NAME.tokens, from ds4_side.sh) to llama-server's /completion as a
token array with temperature 0 and n_probs 5, so both engines see the same
prompt tokens.  It also renders the same messages with the GGUF chat template
(/apply-template) and tokenizes them (/tokenize), recording whether llama.cpp
produces the same ids as ds4.

LLAMA_SERVER names the llama-server binary; LLAMA_ARGS adds server flags
(default "-ngl 99"; "-ngl 0" runs llama.cpp's CPU backend).
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = "You are a helpful assistant"
PORT = 8089


def call(path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    model, tokdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n_predict = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    os.makedirs(out, exist_ok=True)
    server = os.environ.get("LLAMA_SERVER", "llama-server")
    extra = os.environ.get("LLAMA_ARGS", "-ngl 99").split()
    srv = subprocess.Popen(
        [server, "-m", model, "-c", "4096", "--port", str(PORT), "-np", "1", "--no-webui", "--jinja"] + extra,
        stdout=open(os.path.join(out, "server.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(600):
            try:
                if call("/health").get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.5)
        for p in json.load(open(os.path.join(HERE, "prompts.json"))):
            ids = json.loads(open(os.path.join(tokdir, p["name"] + ".tokens")).read())
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["prompt"]}]
            rendered = call("/apply-template", {"messages": messages})["prompt"]
            tokens = call("/tokenize", {"content": rendered, "add_special": False, "parse_special": True})["tokens"]
            res = call("/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0.0,
                                       "n_probs": 5, "cache_prompt": False, "return_tokens": True})
            with open(os.path.join(out, p["name"] + ".json"), "w") as f:
                json.dump({"rendered": rendered, "llama_tokens": tokens, "ds4_tokens": ids, "completion": res},
                          f, ensure_ascii=False, indent=1)
            same = "same prompt tokens" if tokens == ids else "PROMPT TOKENS DIFFER"
            print(p["name"], same, len(res.get("tokens", [])), "tokens", flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=60)


if __name__ == "__main__":
    main()
