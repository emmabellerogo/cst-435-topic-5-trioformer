"""Reproducibly export BERT / GPT-2 / T5 (+ the educational encoder) to quantized ONNX.

Run from the repo root in an environment with the *export* requirements
(``pip install -r requirements-export.txt``); the production server never runs this.

    python scripts/export_onnx.py --out build/models                 # int8 (production)
    python scripts/export_onnx.py --out build/models-fp32 --precision fp32   # exact, for parity tests
    python scripts/export_onnx.py --only edu --out models            # tiny educational encoder

What is produced (all relative to ``--out``)::

    bert/model.onnx[.data]                        BERT masked-LM, returns logits at the [MASK] position
    gpt2/model.onnx[.data]                        GPT-2, one decoding step with a KV cache
    t5-small/{encoder,cross,decoder_step}.onnx    T5 encoder, cross-attention K/V, one decoder step w/ KV cache
    <model>/tokenizer.json                        the checkpoint's own fast-tokenizer file
    edu/block_h{1,2,4}.onnx, edu/embedding.npy    the custom educational encoder (src/transformer.py)
    manifest.json                                 sha256 + size of every file, tool versions, HF revisions

int8 recipe (validated in the investigation): export fp32 -> drop stale value_info -> untie the
shared embedding/output-head weight -> ``quantize_dynamic`` (MatMul+Gemm, per-channel QInt8) ->
write weights as ONE external-data file so onnxruntime memory-maps them instead of copying.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OPSET = 17
CHECKPOINTS = {"bert": "bert-base-uncased", "gpt2": "gpt2", "t5": "t5-small"}
DEFAULT_RELEASE_TAG = "models-v1"
DEFAULT_REPO_SLUG = "emmabellerogo/cst-435-topic-5-trioformer"


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _untie_gather_weights(model) -> None:
    """Give non-Gather consumers of an embedding initializer their own copy.

    BERT/T5 share one tensor between the embedding lookup (Gather) and the output head
    (Gemm/MatMul). The quantizer rewrites the shared tensor for the head and leaves the
    lookup pointing at the wrong thing, so the two uses are separated first.
    """
    inits = {i.name: i for i in model.graph.initializer}
    for name in {n.input[0] for n in model.graph.node if n.op_type == "Gather" and n.input[0] in inits}:
        others = [n for n in model.graph.node if n.op_type != "Gather" and name in n.input]
        if not others:
            continue
        dup = copy.deepcopy(inits[name])
        dup.name = name + "__head"
        model.graph.initializer.append(dup)
        for n in others:
            for k, x in enumerate(n.input):
                if x == name:
                    n.input[k] = dup.name


def finalize_onnx(fp32_path: Path, dst: Path, precision: str) -> None:
    """fp32 ONNX -> (optionally quantized) ONNX with weights in a single external-data file."""
    import onnx

    model = onnx.load(str(fp32_path))
    model.graph.ClearField("value_info")  # stale exporter shape hints break the quantizer
    if precision == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        _untie_gather_weights(model)
        tmp_in, tmp_out = fp32_path.with_suffix(".prep.onnx"), fp32_path.with_suffix(".q.onnx")
        onnx.save(model, str(tmp_in))
        quantize_dynamic(str(tmp_in), str(tmp_out), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"])
        model = onnx.load(str(tmp_out))
    dst.parent.mkdir(parents=True, exist_ok=True)
    for stale in (dst, Path(str(dst) + ".data")):
        if stale.exists():
            stale.unlink()
    onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=dst.name + ".data", size_threshold=1024)


def _revision_and_tokenizer(repo_id: str, out_dir: Path, local_only: bool) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "tokenizer.json", local_files_only=local_only)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, out_dir / "tokenizer.json")
    return Path(path).parent.name  # snapshots/<commit-hash>/tokenizer.json


# --------------------------------------------------------------------------- BERT
def export_bert(out: Path, precision: str, local_only: bool, tmp: Path) -> dict:
    import torch
    from torch import nn
    from transformers import AutoModelForMaskedLM

    class BertMask(nn.Module):  # logits at the mask position only (skips a full-sequence vocab projection)
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids, mask_pos):
            h = self.m.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids).last_hidden_state
            return self.m.cls(h[torch.arange(h.size(0)), mask_pos])

    m = AutoModelForMaskedLM.from_pretrained(CHECKPOINTS["bert"], attn_implementation="eager",
                                             local_files_only=local_only).eval()
    ids = torch.randint(1000, 2000, (1, 9))
    fp32 = tmp / "bert.onnx"
    torch.onnx.export(
        BertMask(m), (ids, torch.ones_like(ids), torch.zeros_like(ids), torch.tensor([4])), str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids", "mask_pos"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "token_type_ids": {0: "b", 1: "s"}, "mask_pos": {0: "b"}, "logits": {0: "b"}},
        opset_version=OPSET)
    finalize_onnx(fp32, out / "bert" / "model.onnx", precision)
    return {"bert-base-uncased": _revision_and_tokenizer(CHECKPOINTS["bert"], out / "bert", local_only)}


# --------------------------------------------------------------------------- GPT-2
def export_gpt2(out: Path, precision: str, local_only: bool, tmp: Path) -> dict:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM

    class Gpt2Step(nn.Module):  # one decoding step; empty past (length 0) == prefill
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, *past):
            pkv = tuple((past[2 * i], past[2 * i + 1]) for i in range(12))
            o = self.m(input_ids=input_ids, past_key_values=pkv, use_cache=True)
            flat = []
            for k, v in o.past_key_values:
                flat += [k, v]
            return (o.logits[:, -1, :], *flat)

    g = AutoModelForCausalLM.from_pretrained(CHECKPOINTS["gpt2"], attn_implementation="eager",
                                             local_files_only=local_only).eval()
    past = [torch.randn(1, 12, 3, 64) for _ in range(24)]
    names_in = ["input_ids"] + [f"past_{i}_{kv}" for i in range(12) for kv in ("k", "v")]
    names_out = ["logits"] + [f"present_{i}_{kv}" for i in range(12) for kv in ("k", "v")]
    dyn = {"input_ids": {1: "s"}, **{n: {2: "p"} for n in names_in[1:]}, **{n: {2: "t"} for n in names_out[1:]}}
    fp32 = tmp / "gpt2.onnx"
    torch.onnx.export(Gpt2Step(g), (torch.randint(0, 50000, (1, 2)), *past), str(fp32),
                      input_names=names_in, output_names=names_out, dynamic_axes=dyn, opset_version=OPSET)
    finalize_onnx(fp32, out / "gpt2" / "model.onnx", precision)
    return {"gpt2": _revision_and_tokenizer(CHECKPOINTS["gpt2"], out / "gpt2", local_only)}


# --------------------------------------------------------------------------- T5
def export_t5(out: Path, precision: str, local_only: bool, tmp: Path) -> dict:
    import torch
    from torch import nn
    from transformers import AutoModelForSeq2SeqLM

    L, H, DH = 6, 8, 64

    class Enc(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    class Cross(nn.Module):  # per-layer cross-attention K/V from encoder output (once per request)
        def __init__(self, m):
            super().__init__()
            self.t = m

        def forward(self, h):
            outs = []
            for blk in self.t.decoder.block:
                a = blk.layer[1].EncDecAttention
                b = h.size(0)
                outs += [a.k(h).view(b, -1, H, DH).transpose(1, 2), a.v(h).view(b, -1, H, DH).transpose(1, 2)]
            return tuple(outs)

    class Step(nn.Module):  # one decoder step with self-attention cache; cross K/V passed in
        def __init__(self, m):
            super().__init__()
            self.t = m

        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask, *past):
            t = self.t
            pkv = tuple(tuple(past[4 * i:4 * i + 4]) for i in range(L))
            o = t.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask, past_key_values=pkv, use_cache=True)
            logits = t.lm_head(o.last_hidden_state * (t.model_dim ** -0.5))[:, -1, :]
            present = []
            for kv in o.past_key_values:
                present += [kv[0], kv[1]]
            return (logits, *present)

    t = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINTS["t5"], local_files_only=local_only).eval()
    assert t.config.tie_word_embeddings
    ei = torch.randint(5, 1000, (1, 11))
    enc_fp32 = tmp / "t5_encoder.onnx"
    torch.onnx.export(Enc(t), (ei, torch.ones_like(ei)), str(enc_fp32), input_names=["input_ids", "attention_mask"],
                      output_names=["hidden"], opset_version=OPSET,
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "hidden": {0: "b", 1: "s"}})
    h = torch.randn(2, 9, 512)
    cross_fp32 = tmp / "t5_cross.onnx"
    torch.onnx.export(Cross(t), (h,), str(cross_fp32), input_names=["hidden"], opset_version=OPSET,
                      output_names=[f"cross_{i}_{kv}" for i in range(L) for kv in "kv"],
                      dynamic_axes={"hidden": {0: "b", 1: "s"}, **{f"cross_{i}_{kv}": {0: "b", 2: "s"} for i in range(L) for kv in "kv"}})
    cross = Cross(t)(h)
    past = []
    for i in range(L):
        past += [torch.randn(2, H, 3, DH), torch.randn(2, H, 3, DH), cross[2 * i], cross[2 * i + 1]]
    names_in = ["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"] + [
        f"past_{i}_{n}" for i in range(L) for n in ("sk", "sv", "ck", "cv")]
    names_out = ["logits"] + [f"present_{i}_{n}" for i in range(L) for n in ("sk", "sv")]
    dyn = {"decoder_input_ids": {0: "b"}, "encoder_hidden_states": {0: "b", 1: "s"},
           "encoder_attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}
    for i in range(L):
        dyn.update({f"past_{i}_sk": {0: "b", 2: "p"}, f"past_{i}_sv": {0: "b", 2: "p"},
                    f"past_{i}_ck": {0: "b", 2: "s"}, f"past_{i}_cv": {0: "b", 2: "s"},
                    f"present_{i}_sk": {0: "b", 2: "q"}, f"present_{i}_sv": {0: "b", 2: "q"}})
    step_fp32 = tmp / "t5_step.onnx"
    torch.onnx.export(Step(t), (torch.zeros(2, 1, dtype=torch.long), h, torch.ones(2, 9, dtype=torch.long), *past),
                      str(step_fp32), input_names=names_in, output_names=names_out, dynamic_axes=dyn,
                      opset_version=OPSET)
    d = out / "t5-small"
    finalize_onnx(enc_fp32, d / "encoder.onnx", precision)
    finalize_onnx(cross_fp32, d / "cross.onnx", precision)
    finalize_onnx(step_fp32, d / "decoder_step.onnx", precision)
    return {"t5-small": _revision_and_tokenizer(CHECKPOINTS["t5"], d, local_only)}


# --------------------------------------------------------------------------- educational encoder
def export_edu(out: Path, precision: str, local_only: bool, tmp: Path) -> dict:
    """The custom encoder block from src/transformer.py with the fixed seed used by the API."""
    import numpy as np
    import torch
    from torch import nn

    from api import encoder_demo as ed

    d = out / "edu"
    d.mkdir(parents=True, exist_ok=True)

    class Block(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return self.b(x)

    for heads in (1, 2, 4):
        emb, _pos, block = ed._model(heads)
        if heads == 4:
            np.save(d / "embedding.npy", emb.weight.detach().numpy().astype(np.float32))
        torch.onnx.export(Block(block), (torch.randn(1, 6, ed.D_MODEL),), str(d / f"block_h{heads}.onnx"),
                          input_names=["x"], output_names=["out", "attn"], opset_version=OPSET,
                          dynamic_axes={"x": {1: "t"}, "out": {1: "t"}, "attn": {2: "t", 3: "t"}})
    return {}


# --------------------------------------------------------------------------- manifest
def write_manifest(out: Path, revisions: dict, precision: str, repo_slug: str, tag: str) -> Path:
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = p.relative_to(out).as_posix()
            files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size,
                          "asset": rel.replace("/", "__"), "bundled": rel.startswith("edu/")}
    manifest = {
        "schema": 1,
        "release_tag": tag,
        "base_url": f"https://github.com/{repo_slug}/releases/download/{tag}",
        "precision": precision,
        "recipe": ("dynamic int8, per-channel QInt8 weights on MatMul/Gemm, shared embedding untied, "
                   "external-data (memory-mapped) weights, opset %d" % OPSET) if precision == "int8"
        else "fp32 export (reference / parity tests only)",
        "tools": {k: ver(k) for k in ("torch", "transformers", "onnx", "onnxruntime", "numpy", "tokenizers")},
        "huggingface_revisions": revisions,
        "files": files,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--only", default="bert,gpt2,t5,edu", help="comma list from: bert,gpt2,t5,edu")
    ap.add_argument("--local-files-only", action="store_true", help="never touch the network (use the HF cache)")
    ap.add_argument("--repo", default=DEFAULT_REPO_SLUG, help="GitHub owner/repo that will host the release assets")
    ap.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    a = ap.parse_args(argv)
    if a.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    torch.manual_seed(0)
    a.out.mkdir(parents=True, exist_ok=True)
    revisions: dict = {}
    funcs = {"bert": export_bert, "gpt2": export_gpt2, "t5": export_t5, "edu": export_edu}
    with tempfile.TemporaryDirectory() as td:
        for name in a.only.split(","):
            print(f"[export] {name} ({a.precision})", flush=True)
            revisions.update(funcs[name](a.out, a.precision if name != "edu" else "fp32", a.local_files_only, Path(td)))
    path = write_manifest(a.out, revisions, a.precision, a.repo, a.release_tag)
    total = sum(f["size"] for f in json.loads(path.read_text())["files"].values())
    print(f"[export] wrote {path} ({total / 1e6:.0f} MB of artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
