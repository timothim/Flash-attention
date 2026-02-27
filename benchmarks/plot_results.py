"""
Generate benchmark plots from saved JSON results.

Produces 5 plots:
  1. Throughput (TFLOPS) vs seq_len — per implementation
  2. Peak memory vs seq_len — linear vs quadratic scaling
  3. Roofline plot — arithmetic intensity vs TFLOPS
  4. Speedup bar chart — relative to naive baseline
  5. CUDA vs Triton direct comparison
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "bench_results.json")

COLORS = {
    "naive": "#e74c3c",
    "flash_cuda": "#2ecc71",
    "pytorch_sdpa": "#3498db",
    "triton": "#f39c12",
    "dao_flash": "#9b59b6",
}

LABELS = {
    "naive": "Naive (materialise N\u00d7N)",
    "flash_cuda": "Flash CUDA (ours)",
    "pytorch_sdpa": "PyTorch SDPA",
    "triton": "Triton (ours)",
    "dao_flash": "Tri Dao flash-attn",
}


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def filter_results(data, **kwargs):
    out = data
    for k, v in kwargs.items():
        out = [r for r in out if r.get(k) == v]
    return out


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Throughput vs seq_len
# ═══════════════════════════════════════════════════════════════════════

def plot_throughput(data, D, causal=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = filter_results(data, D=D, causal=causal)

    impls = sorted(set(r["impl"] for r in subset))
    for impl in impls:
        pts = sorted(filter_results(subset, impl=impl), key=lambda r: r["N"])
        if not pts:
            continue
        ns = [p["N"] for p in pts]
        tflops = [p["tflops"] for p in pts]
        ax.plot(ns, tflops, "o-", color=COLORS.get(impl, "grey"),
                label=LABELS.get(impl, impl), linewidth=2, markersize=6)

    ax.set_xlabel("Sequence Length (N)", fontsize=12)
    ax.set_ylabel("TFLOPS", fontsize=12)
    ax.set_title(f"Throughput vs Sequence Length (D={D}, causal={causal})",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    tag = f"causal" if causal else "full"
    path = os.path.join(RESULTS_DIR, f"throughput_d{D}_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Memory vs seq_len
# ═══════════════════════════════════════════════════════════════════════

def plot_memory(data, D, causal=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = filter_results(data, D=D, causal=causal)

    impls = sorted(set(r["impl"] for r in subset))
    for impl in impls:
        pts = sorted(filter_results(subset, impl=impl), key=lambda r: r["N"])
        if not pts or "peak_mem_mb" not in pts[0]:
            continue
        ns = [p["N"] for p in pts]
        mem = [p["peak_mem_mb"] for p in pts]
        ax.plot(ns, mem, "s-", color=COLORS.get(impl, "grey"),
                label=LABELS.get(impl, impl), linewidth=2, markersize=6)

    ax.set_xlabel("Sequence Length (N)", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title(f"Peak Memory vs Sequence Length (D={D}, causal={causal})",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    tag = f"causal" if causal else "full"
    path = os.path.join(RESULTS_DIR, f"memory_d{D}_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Roofline
# ═══════════════════════════════════════════════════════════════════════

def plot_roofline(data, D, causal=False,
                  peak_tflops=312.0, peak_bw_tb_s=2.0):
    """
    Roofline model for A100:
      peak compute = 312 TFLOPS (FP16 tensor core)
      peak BW      = 2.0 TB/s  (HBM)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = filter_results(data, D=D, causal=causal)

    # Roofline lines.
    ai_range = np.logspace(-1, 4, 200)
    roof = np.minimum(peak_tflops, peak_bw_tb_s * 1e3 * ai_range / 1e3)
    # Correct: roofline = min(peak_compute, peak_bw * AI)
    # Where AI = FLOPs / bytes,  peak_bw in TB/s → GB/s * 1e3
    roof = np.minimum(peak_tflops, peak_bw_tb_s * 1e12 * ai_range / 1e12)
    ax.plot(ai_range, roof, "k--", linewidth=1.5, alpha=0.5, label="Roofline (A100)")

    impls = sorted(set(r["impl"] for r in subset))
    for impl in impls:
        pts = filter_results(subset, impl=impl)
        if not pts:
            continue
        for p in pts:
            B, H, N = p["B"], p["H"], p["N"]
            flops = 4.0 * B * H * N * N * D
            bytes_acc = B * H * (4 * N * D + N) * 2  # FP16
            ai = flops / bytes_acc
            ax.scatter(ai, p["tflops"], color=COLORS.get(impl, "grey"),
                       s=60, alpha=0.7, zorder=5)

        # Single legend entry.
        ax.scatter([], [], color=COLORS.get(impl, "grey"), s=60,
                   label=LABELS.get(impl, impl))

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=12)
    ax.set_ylabel("TFLOPS", fontsize=12)
    ax.set_title(f"Roofline Plot (D={D}, causal={causal})", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    tag = f"causal" if causal else "full"
    path = os.path.join(RESULTS_DIR, f"roofline_d{D}_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Speedup relative to naive
# ═══════════════════════════════════════════════════════════════════════

def plot_speedup(data, D, causal=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = filter_results(data, D=D, causal=causal)

    naive_pts = {r["N"]: r["latency_ms"]
                 for r in filter_results(subset, impl="naive")}
    if not naive_pts:
        print(f"No naive results for D={D} causal={causal}, skipping speedup plot.")
        return

    impls = [i for i in ["flash_cuda", "pytorch_sdpa", "triton", "dao_flash"]
             if any(r["impl"] == i for r in subset)]
    ns = sorted(naive_pts.keys())

    x = np.arange(len(ns))
    width = 0.8 / max(len(impls), 1)

    for i, impl in enumerate(impls):
        speedups = []
        for n in ns:
            impl_pts = filter_results(subset, impl=impl, N=n)
            if impl_pts and n in naive_pts:
                speedups.append(naive_pts[n] / impl_pts[0]["latency_ms"])
            else:
                speedups.append(0)
        ax.bar(x + i * width, speedups, width, color=COLORS.get(impl, "grey"),
               label=LABELS.get(impl, impl))

    ax.set_xlabel("Sequence Length (N)", fontsize=12)
    ax.set_ylabel("Speedup vs Naive", fontsize=12)
    ax.set_title(f"Speedup over Naive Attention (D={D}, causal={causal})",
                 fontsize=14)
    ax.set_xticks(x + width * (len(impls) - 1) / 2)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="grey", linestyle=":", alpha=0.5)

    tag = f"causal" if causal else "full"
    path = os.path.join(RESULTS_DIR, f"speedup_d{D}_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: CUDA vs Triton
# ═══════════════════════════════════════════════════════════════════════

def plot_cuda_vs_triton(data, D, causal=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    subset = filter_results(data, D=D, causal=causal)

    for impl, color in [("flash_cuda", COLORS["flash_cuda"]),
                         ("triton", COLORS["triton"])]:
        pts = sorted(filter_results(subset, impl=impl), key=lambda r: r["N"])
        if not pts:
            continue
        ns = [p["N"] for p in pts]
        ax1.plot(ns, [p["latency_ms"] for p in pts], "o-", color=color,
                 label=LABELS.get(impl, impl), linewidth=2)
        ax2.plot(ns, [p["tflops"] for p in pts], "o-", color=color,
                 label=LABELS.get(impl, impl), linewidth=2)

    ax1.set_xlabel("Sequence Length"); ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency Comparison"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    ax2.set_xlabel("Sequence Length"); ax2.set_ylabel("TFLOPS")
    ax2.set_title("Throughput Comparison"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    fig.suptitle(f"CUDA vs Triton (D={D}, causal={causal})", fontsize=14)
    fig.tight_layout()

    tag = f"causal" if causal else "full"
    path = os.path.join(RESULTS_DIR, f"cuda_vs_triton_d{D}_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}.")
        print("Run `make bench` first.")
        return

    data = load_results()
    print(f"Loaded {len(data)} benchmark entries.\n")

    for D in [64, 128]:
        for causal in [False, True]:
            plot_throughput(data, D, causal)
            plot_memory(data, D, causal)
            plot_roofline(data, D, causal)
            plot_speedup(data, D, causal)
            plot_cuda_vs_triton(data, D, causal)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
