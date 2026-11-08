"""Fair benchmark: our TransformerEncoderBlock vs ``torch.nn.TransformerEncoderLayer``.

Run from the repo root::

    python -m benchmarks.benchmark_encoder            # full run, CPU
    python -m benchmarks.benchmark_encoder --steps 50 --reps 5   # quick

What "fair" means here
----------------------
* Same hyper-parameters: d_model, heads, d_ff, dropout=0, GELU, LayerNorm
  eps=1e-5, pre-norm (``norm_first=True``), ``batch_first=True``.
* Same STARTING WEIGHTS: the torch layer is built with a different random init
  and then every parameter is overwritten from the custom block (the custom
  block's separate Q/K/V weights are concatenated into torch's fused
  ``in_proj_weight`` / ``in_proj_bias``). The embedding, final LayerNorm and
  classifier head are shared by copying too.
* Before anything is trained, forward outputs AND gradients are compared; the
  script aborts if they are not numerically comparable.
* Same data, same batch order, same loss, same optimizer settings, same seeds.
* Timing alternates between implementations to spread machine noise evenly.

Task: the project's palindrome classification (``shared.data``), read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from shared.data import N_CLASSES, VOCAB_SIZE, generate_sequences
from src.transformer import SinusoidalPositionalEncoding, TransformerEncoderBlock

RESULTS_DIR = Path(__file__).parent / "results"
LN_EPS = 1e-5


# ----------------------------------------------------------------- builders
def make_torch_layer(d_model: int, num_heads: int, d_ff: int) -> nn.TransformerEncoderLayer:
    """The reference layer, configured to match our block exactly."""
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=d_ff,
        dropout=0.0,
        activation="gelu",       # exact (erf) GELU, same as nn.GELU()
        layer_norm_eps=LN_EPS,   # same as nn.LayerNorm default
        batch_first=True,
        norm_first=True,         # pre-norm
    )


def make_custom_block(d_model: int, num_heads: int, d_ff: int) -> TransformerEncoderBlock:
    return TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=0.0)


@torch.no_grad()
def copy_custom_to_torch(custom: TransformerEncoderBlock, ref: nn.TransformerEncoderLayer) -> List[str]:
    """Overwrite EVERY parameter of ``ref`` from ``custom``.

    PyTorch fuses the three attention projections into one matrix::

        in_proj_weight [3D, D] = concat(Wq, Wk, Wv)   (rows in q, k, v order)
        in_proj_bias   [3D]    = concat(bq, bk, bv)

    Returns the names of the ``ref`` parameters that were written, so callers
    can check that none were missed.
    """
    a = custom.attn
    ref.self_attn.in_proj_weight.copy_(
        torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0)
    )
    ref.self_attn.in_proj_bias.copy_(
        torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], dim=0)
    )
    ref.self_attn.out_proj.weight.copy_(a.out_proj.weight)
    ref.self_attn.out_proj.bias.copy_(a.out_proj.bias)
    ref.linear1.weight.copy_(custom.mlp[0].weight)
    ref.linear1.bias.copy_(custom.mlp[0].bias)
    ref.linear2.weight.copy_(custom.mlp[2].weight)
    ref.linear2.bias.copy_(custom.mlp[2].bias)
    ref.norm1.weight.copy_(custom.norm1.weight)
    ref.norm1.bias.copy_(custom.norm1.bias)
    ref.norm2.weight.copy_(custom.norm2.weight)
    ref.norm2.bias.copy_(custom.norm2.bias)
    return [
        "self_attn.in_proj_weight", "self_attn.in_proj_bias",
        "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
    ]


class Classifier(nn.Module):
    """embedding + sinusoidal PE -> ONE encoder block -> LayerNorm -> mean-pool -> linear.

    Everything except ``block`` is identical between the two variants.
    """

    def __init__(self, block: nn.Module, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.block = block
        self.norm = nn.LayerNorm(d_model, eps=LN_EPS)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embedding(tokens))
        h = self.block(h)
        if isinstance(h, tuple):          # our block also returns attention weights
            h = h[0]
        return self.head(self.norm(h).mean(dim=1))


@torch.no_grad()
def randomize_layernorms(model: nn.Module, scale: float = 0.1) -> None:
    """Give every LayerNorm a non-default affine (weight != 1, bias != 0).

    By default LayerNorm starts at weight=1 / bias=0 in BOTH implementations, so
    a broken LayerNorm copy would be invisible. Randomising makes the copy
    observable: the equivalence check fails if it is wrong.
    """
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.add_(scale * torch.randn_like(m.weight))
            m.bias.add_(scale * torch.randn_like(m.bias))


def build_pair(
    d_model: int, num_heads: int, d_ff: int, seed: int, max_len: int = 512
) -> Tuple[Classifier, Classifier]:
    """Return ``(custom_model, torch_model)`` with identical starting weights."""
    torch.manual_seed(seed)
    custom = Classifier(make_custom_block(d_model, num_heads, d_ff), d_model, max_len)
    randomize_layernorms(custom)
    torch.manual_seed(seed + 1)  # deliberately DIFFERENT init: proves the copy works
    ref = Classifier(make_torch_layer(d_model, num_heads, d_ff), d_model, max_len)

    written = copy_custom_to_torch(custom.block, ref.block)
    assert set(written) == {n for n, _ in ref.block.named_parameters()}, "unmapped parameter"
    ref.embedding.load_state_dict(custom.embedding.state_dict())
    ref.norm.load_state_dict(custom.norm.state_dict())
    ref.head.load_state_dict(custom.head.state_dict())
    return custom, ref


# ------------------------------------------------------------- equivalence
def equivalence_report(custom: Classifier, ref: Classifier, tokens: torch.Tensor) -> Dict[str, float]:
    """Max abs differences in block output, logits, and (fused-aware) gradients."""
    h = custom.pos(custom.embedding(tokens)).detach()
    report: Dict[str, float] = {}

    # Train-mode forward (dropout=0 so deterministic; grad enabled -> torch's
    # reference path, not its inference fast path).
    custom.train(), ref.train()
    out_c = custom.block(h)[0]
    out_r = ref.block(h)
    report["block_output_max_abs_diff"] = (out_c - out_r).abs().max().item()
    report["block_output_max_abs_value"] = out_c.abs().max().item()

    # Eval + no_grad: torch may take its fused inference fast path here.
    custom.eval(), ref.eval()
    with torch.no_grad():
        report["block_output_eval_fastpath_max_abs_diff"] = (
            custom.block(h)[0] - ref.block(h)
        ).abs().max().item()
        report["logits_max_abs_diff"] = (custom(tokens) - ref(tokens)).abs().max().item()
    custom.train(), ref.train()

    # Gradients: compare after one identical backward pass.
    y = torch.zeros(tokens.size(0), dtype=torch.long)
    for m in (custom, ref):
        m.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(m(tokens), y).backward()
    d = custom.block.attn
    fused_w = torch.cat([d.q_proj.weight.grad, d.k_proj.weight.grad, d.v_proj.weight.grad], 0)
    fused_b = torch.cat([d.q_proj.bias.grad, d.k_proj.bias.grad, d.v_proj.bias.grad], 0)
    g = {
        "in_proj_weight": (fused_w - ref.block.self_attn.in_proj_weight.grad).abs().max(),
        "in_proj_bias": (fused_b - ref.block.self_attn.in_proj_bias.grad).abs().max(),
        "out_proj": (d.out_proj.weight.grad - ref.block.self_attn.out_proj.weight.grad).abs().max(),
        "linear1": (custom.block.mlp[0].weight.grad - ref.block.linear1.weight.grad).abs().max(),
        "linear2": (custom.block.mlp[2].weight.grad - ref.block.linear2.weight.grad).abs().max(),
        "norm1": (custom.block.norm1.weight.grad - ref.block.norm1.weight.grad).abs().max(),
        "embedding": (custom.embedding.weight.grad - ref.embedding.weight.grad).abs().max(),
    }
    report["grad_max_abs_diff"] = max(v.item() for v in g.values())
    custom.zero_grad(set_to_none=True), ref.zero_grad(set_to_none=True)
    return report


# ---------------------------------------------------------------- training
def make_batches(n_rows: int, seq_len: int, batch_size: int, steps: int, seed: int):
    """Deterministic data + a fixed list of batch index tensors shared by both runs."""
    seqs, labels = generate_sequences(n_rows, seq_len=seq_len, seed=seed)
    x = torch.from_numpy(seqs)
    y = torch.tensor(labels, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    order: List[torch.Tensor] = []
    while len(order) < steps:
        perm = torch.randperm(n_rows, generator=g)
        order += [perm[i : i + batch_size] for i in range(0, n_rows - batch_size + 1, batch_size)]
    return x, y, order[:steps]


def train(model: Classifier, x, y, batches, lr: float) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    history = []
    for idx in batches:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


# ------------------------------------------------------------------ timing
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class _Timer:
    """Holds one model + optimizer + fixed batch; ``rep()`` times a block of steps."""

    def __init__(self, model: Classifier, xb, yb, device: torch.device, lr: float, warmup: int):
        self.model, self.xb, self.yb, self.device = model.to(device), xb.to(device), yb.to(device), device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(warmup):
            self._step()
        _sync(device)

    def _step(self) -> None:
        self.opt.zero_grad(set_to_none=True)
        self.loss_fn(self.model(self.xb), self.yb).backward()
        self.opt.step()

    def rep(self, steps: int) -> float:
        _sync(self.device)
        t0 = time.perf_counter()
        for _ in range(steps):
            self._step()
        _sync(self.device)
        return (time.perf_counter() - t0) * 1000.0 / steps


def benchmark_timing(
    cfg: Dict[str, int], device: torch.device, warmup: int, steps_per_rep: int, reps: int, lr: float, seed: int
) -> Dict[str, Dict[str, float]]:
    custom, ref = build_pair(cfg["d_model"], cfg["heads"], cfg["d_ff"], seed, max_len=max(512, cfg["seq_len"]))
    g = torch.Generator().manual_seed(seed)
    xb = torch.randint(1, VOCAB_SIZE, (cfg["batch"], cfg["seq_len"]), generator=g)
    yb = torch.randint(0, N_CLASSES, (cfg["batch"],), generator=g)
    timers = {
        "custom": _Timer(custom, xb, yb, device, lr, warmup),
        "torch": _Timer(ref, xb, yb, device, lr, warmup),
    }
    samples: Dict[str, List[float]] = {"custom": [], "torch": []}
    for r in range(reps):  # alternate who goes first to cancel order/thermal bias
        order = ("custom", "torch") if r % 2 == 0 else ("torch", "custom")
        for name in order:
            samples[name].append(timers[name].rep(steps_per_rep))
    out = {}
    for name, v in samples.items():
        out[name] = {
            "median_ms": statistics.median(v),
            "mean_ms": statistics.fmean(v),
            "min_ms": min(v),
            "stdev_ms": statistics.stdev(v) if len(v) > 1 else 0.0,
            "samples_ms": v,
        }
    out["ratio_custom_over_torch_median"] = out["custom"]["median_ms"] / out["torch"]["median_ms"]
    return out


def hardware_info(device: torch.device) -> Dict[str, object]:
    cpu = ""
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return {
        "cpu": cpu or platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
    }


# -------------------------------------------------------------------- main
def run(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, seq_len=args.seq_len, batch=args.batch_size)

    # 1) equivalence gate ------------------------------------------------
    custom, ref = build_pair(args.d_model, args.heads, args.d_ff, args.seed)
    x, y, batches = make_batches(args.n_rows, args.seq_len, args.batch_size, args.steps, args.seed)
    rep = equivalence_report(custom, ref, x[:16])
    print("Equivalence before training:")
    for k, v in rep.items():
        print(f"  {k:45s} {v:.3e}")
    if rep["block_output_max_abs_diff"] > args.tol or rep["grad_max_abs_diff"] > args.tol:
        raise SystemExit(f"Blocks are not numerically equivalent (tol={args.tol}); refusing to benchmark.")

    # 2) training-loss comparison (CPU, deterministic) ----------------------
    h_custom = train(custom, x, y, batches, args.lr)
    h_ref = train(ref, x, y, batches, args.lr)
    diffs = np.abs(np.array(h_custom) - np.array(h_ref))

    # float32 training is chaotic: ~1e-7 rounding differences get amplified over
    # hundreds of Adam steps, so two float32 runs of the SAME math can drift
    # apart. A float64 rerun (rounding ~1e-16) shows the implementations agree.
    c64, r64 = build_pair(args.d_model, args.heads, args.d_ff, args.seed)
    c64, r64 = c64.double(), r64.double()
    h_custom64 = train(c64, x, y, batches, args.lr)
    h_ref64 = train(r64, x, y, batches, args.lr)
    diffs64 = np.abs(np.array(h_custom64) - np.array(h_ref64))

    # Multi-seed: is final loss statistically the same? (mean of last 30 steps)
    tail = min(30, args.steps)
    seed_rows = []
    for sd in range(args.seed, args.seed + args.n_seeds):
        cs, rs = build_pair(args.d_model, args.heads, args.d_ff, sd)
        xs, ys, bs = make_batches(args.n_rows, args.seq_len, args.batch_size, args.steps, sd)
        seed_rows.append({
            "seed": sd,
            "custom_final": float(np.mean(train(cs, xs, ys, bs, args.lr)[-tail:])),
            "torch_final": float(np.mean(train(rs, xs, ys, bs, args.lr)[-tail:])),
        })

    csv_path = RESULTS_DIR / "loss_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "custom_loss_f32", "torch_loss_f32", "abs_diff_f32",
                    "custom_loss_f64", "torch_loss_f64", "abs_diff_f64"])
        for i in range(args.steps):
            w.writerow([i + 1,
                        f"{h_custom[i]:.8f}", f"{h_ref[i]:.8f}", f"{diffs[i]:.3e}",
                        f"{h_custom64[i]:.10f}", f"{h_ref64[i]:.10f}", f"{diffs64[i]:.3e}"])
    plot_losses(h_custom, h_ref, h_custom64, h_ref64, RESULTS_DIR / "loss_plot.png", args)

    # 3) timing ---------------------------------------------------------------
    timing = {}
    configs = {"small (training config)": cfg}
    if not args.skip_large:
        configs["large (scaling check)"] = dict(d_model=256, heads=8, d_ff=1024, seq_len=128, batch=32)
    for label, c in configs.items():
        timing[label] = {"config": c, **benchmark_timing(c, device, args.warmup, args.timing_steps, args.reps, args.lr, args.seed)}

    result = {
        "hardware": hardware_info(device),
        "method": {
            "warmup_steps": args.warmup, "steps_per_rep": args.timing_steps, "reps": args.reps,
            "step": "zero_grad + forward + cross-entropy + backward + Adam.step on a fixed batch",
            "clock": "time.perf_counter around a block of steps; device synchronized before/after (no-op on CPU)",
            "ordering": "implementations alternate order every rep",
        },
        "equivalence_before_training": rep,
        "training": {
            "config": cfg, "steps": args.steps, "lr": args.lr, "seed": args.seed,
            "final_loss_custom": h_custom[-1], "final_loss_torch": h_ref[-1],
            "first_loss_custom": h_custom[0], "first_loss_torch": h_ref[0],
            "max_abs_loss_diff": float(diffs.max()), "mean_abs_loss_diff": float(diffs.mean()),
            "float64_max_abs_loss_diff": float(diffs64.max()),
            "float64_final_loss_custom": h_custom64[-1], "float64_final_loss_torch": h_ref64[-1],
            "multi_seed": {
                "tail_steps_averaged": tail, "per_seed": seed_rows,
                "custom_mean": float(np.mean([r["custom_final"] for r in seed_rows])),
                "custom_std": float(np.std([r["custom_final"] for r in seed_rows])),
                "torch_mean": float(np.mean([r["torch_final"] for r in seed_rows])),
                "torch_std": float(np.std([r["torch_final"] for r in seed_rows])),
            },
        },
        "timing": timing,
    }
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print_summary(result)
    return result


def plot_losses(h_c, h_r, h_c64, h_r64, path: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(h_c) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax1.plot(steps, h_c, label="Custom TransformerEncoderBlock (float32)", lw=2, color="tab:blue")
    ax1.plot(steps, h_r, label="torch.nn.TransformerEncoderLayer (float32)", lw=1.5, ls="--", color="tab:orange")
    ax1.plot(steps, h_c64, label="Custom (float64)", lw=1, color="tab:green", alpha=0.8)
    ax1.plot(steps, h_r64, label="torch (float64)", lw=1, ls=":", color="k", alpha=0.8)
    ax1.set_ylabel("Training loss (cross-entropy)")
    ax1.set_title(f"Palindrome task, identical init/data/optimizer (Adam lr={args.lr}, "
                  f"batch={args.batch_size}, d_model={args.d_model})", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.semilogy(steps, np.abs(np.array(h_c) - np.array(h_r)) + 1e-17, label="float32 |custom - torch|", color="tab:red")
    ax2.semilogy(steps, np.abs(np.array(h_c64) - np.array(h_r64)) + 1e-17, label="float64 |custom - torch|", color="tab:green")
    ax2.set_ylabel("|loss difference|")
    ax2.set_xlabel("Training step")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(r: Dict[str, object]) -> None:
    t = r["training"]
    print(f"\nTraining ({t['steps']} steps): first loss custom={t['first_loss_custom']:.4f} torch={t['first_loss_torch']:.4f}; "
          f"final custom={t['final_loss_custom']:.4f} torch={t['final_loss_torch']:.4f}; "
          f"max |diff|={t['max_abs_loss_diff']:.2e}")
    print(f"float64 rerun: final custom={t['float64_final_loss_custom']:.4f} torch={t['float64_final_loss_torch']:.4f}; "
          f"max |diff|={t['float64_max_abs_loss_diff']:.2e}")
    m = t["multi_seed"]
    print(f"{len(m['per_seed'])} seeds, final loss (mean of last {m['tail_steps_averaged']} steps): "
          f"custom {m['custom_mean']:.4f} +/- {m['custom_std']:.4f}, torch {m['torch_mean']:.4f} +/- {m['torch_std']:.4f}")
    hw = r["hardware"]
    print(f"\nTiming on {hw['cpu']} ({hw['device']}, {hw['torch_threads']} threads, torch {hw['torch']})")
    for label, d in r["timing"].items():
        print(f"  {label}: {d['config']}")
        for name in ("custom", "torch"):
            s = d[name]
            print(f"    {name:7s} median {s['median_ms']:8.3f} ms/step  min {s['min_ms']:8.3f}  sd {s['stdev_ms']:.3f}")
        print(f"    custom / torch (median) = {d['ratio_custom_over_torch_median']:.2f}x")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--n-rows", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=5, help="seeds for the multi-seed final-loss check")
    p.add_argument("--steps", type=int, default=300, help="training steps for the loss comparison")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timing-steps", type=int, default=30, help="steps per timed repetition")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--tol", type=float, default=1e-4, help="max allowed abs diff in the equivalence gate")
    p.add_argument("--skip-large", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
