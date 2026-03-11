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

DEFAULT_OUT_PNG = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/figures/boxFigure/box_metrics_iou0.3.png"
DEFAULT_OUT_PDF = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/figures/boxFigure/box_metrics_iou0.3.pdf"

DEFAULT_DPI = 600
DEFAULT_STYLE = "seaborn-v0_8-whitegrid"

# Organelle order (5)
ORGANELLES = ["er", "lipidDroplet", "lysosome", "mitochondria", "nuclei"]

# Accept legacy keys in metrics JSONs
ORGANELLE_ALIASES = {
    "mitochondria": ["mitochondria", "mito"],
}

# Colors tuned to a vivid, paper-friendly palette (inspired by the reference figure)
PALETTE = {
    "er": "#ff00ff",  # magenta
    "lipidDroplet": "#00ffff",  # cyan
    "lysosome": "#0000ff",  # blue
    "mitochondria": "#ff0000",  # red
    "nuclei": "#ffa500",  # orange
}

METRICS = [
    ("Precision", ["precision"]),
    ("Recall", ["recall"]),
    ("F1 Score", ["f1"]),
    ("Boundary F1", ["boundary_f1", "f1"]),
]


def _extract_slice_from_path(p: str) -> int | None:
    m = re.search(r"metrics_(\d+)_iou", p)
    return int(m.group(1)) if m else None


def _safe_get(d: dict, keys: list[str], default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _set_rcparams():
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
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 0.8,
        }
    )


def _load_metrics(path: str) -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["__path__"] = str(p)
    data["__slice__"] = _extract_slice_from_path(str(p))
    return data


def _collect_values(datas: list[dict]) -> dict:
    """Return values[metric_name][organelle] = list of values (one per slice)."""
    values: dict[str, dict[str, list[float]]] = {}
    for metric_name, _ in METRICS:
        values[metric_name] = {o: [] for o in ORGANELLES}

    for d in datas:
        pc = d.get("per_class", {}) or {}
        for organelle in ORGANELLES:
            aliases = ORGANELLE_ALIASES.get(organelle, [organelle])
            v = None
            for key in aliases:
                v = pc.get(key)
                if v:
                    break

            if not v:
                # missing class for this slice
                for metric_name, _ in METRICS:
                    values[metric_name][organelle].append(np.nan)
                continue

            for metric_name, key_path in METRICS:
                val = _safe_get(v, key_path, default=np.nan)
                try:
                    val = float(val)
                except Exception:
                    val = np.nan
                values[metric_name][organelle].append(val)

    return values


def plot_box_figure(
    metrics_paths: list[str],
    out_png: str,
    out_pdf: str | None = None,
    title: str | None = None,
) -> None:
    datas = [_load_metrics(p) for p in metrics_paths]

    # Keep slice ordering increasing
    slices = [d.get("__slice__") for d in datas]
    order = np.argsort([s if s is not None else 1e9 for s in slices])
    datas = [datas[i] for i in order]

    values = _collect_values(datas)

    # Style
    if DEFAULT_STYLE in plt.style.available:
        plt.style.use(DEFAULT_STYLE)
    _set_rcparams()

    # Layout: single axis; each metric is one adjacent group (no gaps between metrics)
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 3.2))

    ax.set_ylim(0.0, 1.0)
    # seaborn-v0_8-whitegrid enables x-grid by default; disable x-grid explicitly.
    ax.grid(True, axis="y", linestyle=(0, (1, 4)), linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")

    # Remove vertical spines so there is no solid vertical line in the middle
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Increase spacing between metric groups (Precision/Recall/...) while keeping boxes within each group.
    group_gap = 1
    group_positions = np.arange(len(METRICS), dtype=float) * group_gap

    # Dashed separators between metric groups
    for x in (group_positions[:-1] + group_positions[1:]) / 2:
        ax.axvline(x, linestyle=(0, (1, 4)), linewidth=1.0, color="0.4", alpha=0.8, zorder=0)

    # Make groups contiguous: keep group spacing at 1.0, and fit organelle boxes within each group.
    # Total span of organelles inside a group ~= 0.9
    # Make boxes slimmer/taller-looking by reducing their width and tightening intra-group spacing.
    box_width = 0.10
    offsets = np.linspace(-2, 2, len(ORGANELLES)) * (box_width * 1.60)

    # Draw per-metric per-organelle boxes
    for m_idx, (metric_name, _) in enumerate(METRICS):
        for o_idx, organelle in enumerate(ORGANELLES):
            vals = np.array(values[metric_name][organelle], dtype=float)
            vals = vals[~np.isnan(vals)]

            pos = group_positions[m_idx] + offsets[o_idx]

            if vals.size == 0:
                continue

            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
            )

            color = PALETTE.get(organelle, "#333333")

            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
                patch.set_edgecolor(color)
                patch.set_linewidth(1.0)

            for med in bp["medians"]:
                med.set_color(color)
                med.set_linewidth(2.0)

            for w in bp["whiskers"]:
                w.set_color(color)
                w.set_linewidth(1.2)

            for c in bp["caps"]:
                c.set_color(color)
                c.set_linewidth(1.2)

            # Overlay mean point
            ax.scatter(
                [pos],
                [float(np.mean(vals))],
                s=18,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    ax.set_xticks(group_positions)
    ax.set_xticklabels([m[0] for m in METRICS])

    ax.set_ylabel("Score")

    # Legend (organelle colors)
    handles = [
        mpl.patches.Patch(
            facecolor=PALETTE[o],
            edgecolor=PALETTE[o],
            alpha=0.55,
            label=("ER" if o == "er" else "mitochondria" if o == "mitochondria" else o),
        )
        for o in ORGANELLES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
        labelspacing=0.2,
    )

    if title:
        fig.suptitle(title, y=1.12)

    fig.tight_layout()

    out_png = str(Path(out_png))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_DPI, bbox_inches="tight")

    if out_pdf:
        out_pdf = str(Path(out_pdf))
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)


def main() -> int:
    # print fonts
    print("===  matplotlib fonts ===")
    print(f"font.family = {mpl.rcParams['font.family']}")
    print(f"font.sans-serif (first 10) = {mpl.rcParams['font.sans-serif'][:10]}")
    print(f"font.size = {mpl.rcParams['font.size']}")
    
    # fonts
    try:
        from matplotlib import font_manager as fm
        from matplotlib.font_manager import FontProperties
        prop = FontProperties(family=mpl.rcParams["font.family"])
        default_font = fm.findfont(prop)
        print(f"fonts = {default_font}")
    except Exception as e:
        print(f"fonts error: {e}")
    
    print("=============================")

    parser = argparse.ArgumentParser(
        description=(
            "Plot box figures for Instance Precision/Recall/F1 and Boundary F1 across slices, grouped by organelle."
        )
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--out_png", type=str, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out_pdf", type=str, default=DEFAULT_OUT_PDF)
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    plot_box_figure(args.metrics, args.out_png, args.out_pdf, title=args.title)

    print("wrote", args.out_png)
    if args.out_pdf:
        print("wrote", args.out_pdf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
