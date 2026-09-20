# Same weights, two configs: canonical Mellum (layer-selective YaRN) vs the
# flattened qwen3_moe rope the old local snapshot shipped. Quantifies what the
# corrected config buys, independent of llama.cpp.
import json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

REPO = "JetBrains/sft_mellum_v23_mixopt_fullx5_p2_joint-iter-3015"
PROMPT = ("<|im_start|>user\nComplete this Python function: def add(a, b):<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
tok = AutoTokenizer.from_pretrained(REPO)
ids = torch.tensor([tok(PROMPT, add_special_tokens=False)["input_ids"]])

def run(cfg, tag):
    m = AutoModelForCausalLM.from_pretrained(REPO, config=cfg, dtype=torch.float32, device_map="cpu").eval()
    caught = {}
    h = m.model.layers[27].register_forward_hook(
        lambda mod, i, out: caught.__setitem__(27, (out[0] if isinstance(out, tuple) else out).detach()[0].float().numpy()))
    with torch.no_grad():
        o = m(ids, use_cache=False)
    h.remove()
    r = (caught[27][-1], o.logits[0, -1].float().numpy())
    print(tag, "argmax", int(r[1].argmax()), repr(tok.decode([int(r[1].argmax())])))
    del m
    return r

cfg = AutoConfig.from_pretrained(REPO)
h_mellum, l_mellum = run(cfg, "mellum   ")

flat = AutoConfig.from_pretrained(REPO)
# Flatten: YaRN everywhere, as the qwen3_moe export expressed it.
yarn = dict(flat.rope_parameters["full_attention"])
flat.rope_parameters = {"full_attention": yarn, "sliding_attention": dict(yarn)}
h_flat, l_flat = run(flat, "flat-yarn")

for tag, a, b in [("l_out-27 (pre-norm)", h_mellum, h_flat), ("logits", l_mellum, l_flat)]:
    d = (a.astype(np.float64) - b.astype(np.float64))
    ref = np.sqrt((b.astype(np.float64) ** 2).mean())
    print(f"{tag:18s} max {np.abs(d).max():12.6g}  RMS {np.sqrt((d**2).mean()):12.6g}  rel {np.sqrt((d**2).mean())/ref:.4%}")
t1 = np.argsort(-l_mellum)[:16].tolist(); t2 = np.argsort(-l_flat)[:16].tolist()
print("top16 identical:", t1 == t2, "| overlap", len(set(t1) & set(t2)), "/16")
json.dump({"mellum_top16": t1, "flat_top16": t2}, open("rope_ablation.json", "w"), indent=1)
