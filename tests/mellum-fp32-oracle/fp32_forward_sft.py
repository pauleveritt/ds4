import hashlib, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "JetBrains/sft_mellum_v23_mixopt_fullx5_p2_joint-iter-3015"  # private JetBrains snapshot
VEC = "/Users/pauleveritt/projects/ds4/.claude/worktrees/mellum-2.1/tests/test-vectors/mellum-llama-cpp"
PROMPT = ("<|im_start|>user\nComplete this Python function: def add(a, b):<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")

def load(name):
    b = open(f"{VEC}/{name}", "rb").read()
    return np.frombuffer(b, "<f4").astype(np.float64), hashlib.sha256(b).hexdigest()

def cmp(tag, a, b):
    a = np.asarray(a, np.float64).ravel(); b = b.ravel(); d = a - b
    rms = np.sqrt((d**2).mean()); ref = np.sqrt((b**2).mean())
    cos = (a@b)/(np.linalg.norm(a)*np.linalg.norm(b))
    print(f"{tag:30s} max {np.abs(d).max():11.5g}  RMS {rms:11.5g}  rel {rms/ref:8.2%}  cos {cos:+.6f}")

tok = AutoTokenizer.from_pretrained(REPO)
ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32, device_map="cpu").eval()

caught = {}
def hook(i):
    def f(mod, inp, out):
        caught[i] = (out[0] if isinstance(out, tuple) else out).detach()[0].float().numpy()
    return f
hs = [model.model.layers[i].register_forward_hook(hook(i)) for i in (0, 27)]
with torch.no_grad():
    o = model(torch.tensor([ids]), use_cache=False)
for h in hs: h.remove()

l0, l27 = caught[0], caught[27]
logits = o.logits[0, -1].float().numpy()
np.save("thinking_l0_prenorm.npy", l0); np.save("thinking_l27_prenorm.npy", l27); np.save("thinking_logits.npy", logits)

ref0, _ = load("l-out-0.f32"); ref27, _ = load("l-out-27-tokenwise.f32"); refL, _ = load("result-output-tokenwise.f32")
print(f"\nlayer-0   RMS  HF {np.sqrt((l0.astype(np.float64)**2).mean()):9.4f}   ref {np.sqrt((ref0**2).mean()):9.4f}")
print(f"layer-27  RMS  HF {np.sqrt((l27[-1].astype(np.float64)**2).mean()):9.4f}   ref {np.sqrt((ref27**2).mean()):9.4f}"
      f"   max|x| HF {np.abs(l27[-1]).max():.1f} ref {np.abs(ref27).max():.1f}")
print(f"logits    RMS  HF {np.sqrt((logits.astype(np.float64)**2).mean()):9.4f}   ref {np.sqrt((refL**2).mean()):9.4f}\n")
cmp("l_out-0 (26x2304)", l0, ref0)
cmp("l_out-27 (last row, pre-norm)", l27[-1], ref27)
cmp("result_output logits", logits, refL)

t1 = np.argsort(-logits)[:16].tolist(); t2 = np.argsort(-refL)[:16].tolist()
print("\nHF top16 :", t1, "->", repr(tok.decode([t1[0]])))
print("ref top16:", t2, "->", repr(tok.decode([t2[0]])))
print("overlap", len(set(t1) & set(t2)), "/16  identical:", t1 == t2)
json.dump({"ids": ids, "hf_top16": t1, "ref_top16": t2}, open("fp32_thinking_summary.json", "w"), indent=1)
