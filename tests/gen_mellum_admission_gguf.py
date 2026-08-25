#!/usr/bin/env python3
"""Emit a sparse, single-layer (layer 1) Mellum GGUF for the admission test.

--down-type q8_0 is the valid fixture; q5_0 is the invalid one the loader must
refuse. Only header + metadata + tensor directory are written; the payload is a
sparse hole (never read during admission), so the file is ~Kib on disk.
"""
import argparse
import struct

MAGIC = 0x46554747          # "GGUF", little-endian
VERSION = 3
ALIGN = 32

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
F32, Q8_0, Q5_0 = 0, 8, 6


def s(b):
    data = b.encode() if isinstance(b, str) else b
    return struct.pack("<Q", len(data)) + data


def kv(key, vtype, payload):
    return s(key) + struct.pack("<I", vtype) + payload


def p_u32(v):
    return struct.pack("<I", v)


def p_u64(v):
    return struct.pack("<Q", v)


def p_f32(v):
    return struct.pack("<f", v)


def tensor(name, ndim, dims, ttype):
    return (s(name) + struct.pack("<I", ndim) +
            b"".join(struct.pack("<Q", d) for d in dims) +
            struct.pack("<I", ttype) + struct.pack("<Q", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--down-type", choices=("q8_0", "q5_0"), required=True)
    a = ap.parse_args()
    down_type = Q8_0 if a.down_type == "q8_0" else Q5_0

    pattern = bytes(1 if ((i & 3) != 3) else 0 for i in range(28))

    meta = [
        kv("general.architecture", STRING, s("mellum")),
        kv("general.alignment", UINT32, p_u32(ALIGN)),
        kv("mellum.block_count", UINT32, p_u32(28)),
        kv("mellum.context_length", UINT64, p_u64(131072)),
        kv("mellum.embedding_length", UINT32, p_u32(2304)),
        kv("mellum.feed_forward_length", UINT32, p_u32(7168)),
        kv("mellum.attention.head_count", UINT32, p_u32(32)),
        kv("mellum.attention.head_count_kv", UINT32, p_u32(4)),
        kv("mellum.attention.key_length", UINT32, p_u32(128)),
        kv("mellum.attention.value_length", UINT32, p_u32(128)),
        kv("mellum.expert_count", UINT32, p_u32(64)),
        kv("mellum.expert_used_count", UINT32, p_u32(8)),
        kv("mellum.expert_feed_forward_length", UINT32, p_u32(896)),
        kv("mellum.attention.sliding_window", UINT32, p_u32(1024)),
        kv("mellum.attention.sliding_window_pattern", ARRAY,
           struct.pack("<I", BOOL) + struct.pack("<Q", 28) + pattern),
        kv("mellum.rope.scaling.type", STRING, s("yarn")),
        kv("mellum.rope.scaling.original_context_length", UINT64, p_u64(8192)),
        kv("mellum.rope.scaling.factor", FLOAT32, p_f32(16.0)),
        kv("mellum.rope.scaling.yarn_attn_factor", FLOAT32, p_f32(1.2772589)),
        kv("mellum.rope.scaling.yarn_beta_fast", FLOAT32, p_f32(32.0)),
        kv("mellum.rope.scaling.yarn_beta_slow", FLOAT32, p_f32(1.0)),
        kv("mellum.rope.freq_base", FLOAT32, p_f32(500000.0)),
        kv("mellum.rope.freq_base_swa", FLOAT32, p_f32(500000.0)),
        kv("mellum.attention.layer_norm_rms_epsilon", FLOAT32, p_f32(1.0e-6)),
    ]

    tensors = [
        tensor("blk.1.attn_norm.weight", 1, [2304], F32),
        tensor("blk.1.attn_q.weight", 2, [2304, 4096], Q8_0),
        tensor("blk.1.attn_q_norm.weight", 1, [128], F32),
        tensor("blk.1.attn_k.weight", 2, [2304, 512], Q8_0),
        tensor("blk.1.attn_k_norm.weight", 1, [128], F32),
        tensor("blk.1.attn_v.weight", 2, [2304, 512], Q8_0),
        tensor("blk.1.attn_output.weight", 2, [4096, 2304], Q8_0),
        tensor("blk.1.ffn_norm.weight", 1, [2304], F32),
        tensor("blk.1.ffn_gate_inp.weight", 2, [2304, 64], F32),
        tensor("blk.1.ffn_gate_exps.weight", 3, [2304, 896, 64], Q8_0),
        tensor("blk.1.ffn_up_exps.weight", 3, [2304, 896, 64], Q8_0),
        tensor("blk.1.ffn_down_exps.weight", 3, [896, 2304, 64], down_type),
    ]

    header = (struct.pack("<I", MAGIC) + struct.pack("<I", VERSION) +
              struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta)))
    body = b"".join(meta) + b"".join(tensors)
    dir_end = len(header) + len(body)
    data_pos = (dir_end + ALIGN - 1) // ALIGN * ALIGN

    # Largest tensor byte span (gate/up/down, 2304*896*64 elements). Q8_0 packs
    # 32 elements per 34 bytes -> ~134 MiB; Q5_0 packs 32 per 22 bytes -> ~87 MiB.
    # Over-cover with 2x (sparse, so disk cost is negligible).
    max_bytes = 2304 * 896 * 64 * 2
    total = data_pos + max_bytes

    with open(a.out, "wb") as f:
        f.write(header)
        f.write(body)
        f.write(b"\x00" * (data_pos - dir_end))
        f.truncate(total)
    print(f"wrote {a.out} ({total} logical bytes, sparse)")


if __name__ == "__main__":
    main()
