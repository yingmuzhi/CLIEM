from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Plotting requires matplotlib in env_cp311_ymz. Please install it first."
    ) from e


DEFAULT_METRICS = [
    "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/metrics_244_iou0.3.json",
    "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/metrics_288_iou0.3.json",
    "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/metrics_320_iou0.3.json",
    "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/metrics_348_iou0.3.json",
]

DEFAULT_OUT_PNG = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/figures/confusionFigure/confusion_summary_iou0.3.png"
DEFAULT_OUT_PDF = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/figures/confusionFigure/confusion_summary_iou0.3.pdf"

DEFAULT_DPI = 600
DEFAULT_STYLE = "seaborn-v0_8-whitegrid"

ORG_ORDER = ["ER", "lipidDroplet", "lysosome", "mitochondria", "nuclei"]
METRICS = [
    ("Precision", ["precision"]),
    ("Recall", ["recall"]),
    ("F1 Score", ["f1"]),
    ("Boundary F1", ["boundary_f1", "f1"]),
]


def _slice_from_path(p: str) -> int | None:
    m = re.search(r"metrics_(\d+)_iou", p)
    return int(m.group(1)) if m else None


def _safe_get(d: dict, keys: list[str], default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_metrics(paths: list[str]) -> list[dict]:
    datas: list[dict] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        data["__path__"] = p
        data["__slice__"] = _slice_from_path(p)
        datas.append(data)

    datas.sort(key=lambda d: d["__slice__"] if isinstance(d.get("__slice__"), int) else 10**9)
    return datas


def _build_metric_matrix(datas: list[dict], key_path: list[str]) -> tuple[list[int], np.ndarray]:
    slices: list[int] = []
    mat = np.full((len(ORG_ORDER), len(datas)), np.nan, dtype=float)

    for j, d in enumerate(datas):
        s = d.get("__slice__")
        slices.append(int(s) if s is not None else j)
        pc = d.get("per_class", {}) or {}
        for i, organelle in enumerate(ORG_ORDER):
            key = "er" if organelle == "ER" else ("mito" if organelle == "mitochondria" else organelle)
            v = pc.get(key)
            if not v:
                continue
            mat[i, j] = float(_safe_get(v, key_path, default=np.nan))

    return slices, mat


def plot_confusion_merged(
    datas: list[dict],
    out_png: str,
    out_pdf: str | None = None,
    title: str | None = None,
) -> None:
    if DEFAULT_STYLE in plt.style.available:
        plt.style.use(DEFAULT_STYLE)

    mpl.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    mpl.rcParams.update(
        {
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.linewidth": 0.8,
        }
    )

    cmap = mpl.colormaps.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    # Build each metric matrix (organelle x slice)
    slices = None
    mats: list[np.ndarray] = []
    for metric_title, key_path in METRICS:
        s, mat = _build_metric_matrix(datas, key_path)
        if slices is None:
            slices = s
        mats.append(mat)

    assert slices is not None
    n_slice = len(slices)
    n_metrics = len(METRICS)

    # Concatenate horizontally: [Precision|Recall|F1|BoundaryF1], each block has n_slice columns
    big = np.concatenate(mats, axis=1)  # shape: (n_org, n_metrics*n_slice)

    # Figure layout: big heatmap + 4 colorbars at the right
    fig = plt.figure(figsize=(10.8, 3.8))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.18],
        wspace=0.1,
    )

    ax = fig.add_subplot(gs[0, 0])

    # Heatmap
    im = ax.imshow(big, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)

    # y-axis organelles
    ax.set_yticks(np.arange(len(ORG_ORDER)))
    ax.set_yticklabels(ORG_ORDER)
    ax.set_ylabel("Organelle")

    # x-axis ticks: show slice labels repeated per metric block
    xticks = []
    xlabels = []
    for m in range(n_metrics):
        for j in range(n_slice):
            xticks.append(m * n_slice + j)
            xlabels.append(str(j + 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Slice")

    # Remove cell boundary gridlines to show only color blocks
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.grid(False)

    # Thick separators between metric blocks
    for m in range(1, n_metrics):
        ax.axvline(m * n_slice - 0.5, color="#404040", linewidth=1.5)

    # Metric titles centered over each block
    for m, (metric_title, _) in enumerate(METRICS):
        center = m * n_slice + (n_slice - 1) / 2.0
        ax.text(
            center,
            -0.9,
            metric_title,
            ha="center",
            va="bottom",
            transform=ax.transData,
        )

    # Single horizontal colorbar (one shared value gradient)
    # Keep the layout grid but hide the right column axis.
    cax_container = fig.add_subplot(gs[0, 1])
    cax_container.axis("off")

    cb_ax = fig.add_axes([0.92, 0.14, 0.02, 0.72])
    cb = fig.colorbar(im, cax=cb_ax, orientation="vertical")
    cb.set_label("value")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=mpl.rcParams["ytick.labelsize"])

    if title:
        fig.suptitle(title, y=1.03)

    out_png = str(Path(out_png))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_DPI, bbox_inches="tight")

    if out_pdf:
        out_pdf = str(Path(out_pdf))
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a merged heatmap figure: y=organelle, x=(metric blocks × slices), with 4 colorbars on the right."
        )
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--out_png", type=str, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out_pdf", type=str, default=DEFAULT_OUT_PDF)
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    datas = _load_metrics(args.metrics)
    plot_confusion_merged(datas, args.out_png, args.out_pdf, title=args.title)

    print("wrote", args.out_png)
    if args.out_pdf:
        print("wrote", args.out_pdf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
