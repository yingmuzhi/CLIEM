#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import Dict, List

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricRow:
    name: str
    x_label: str
    # five organelle fractions (percent) used for stacked bar, in the same order as `organelles`
    fractions: Dict[str, List[float]]
    # The absolute total value for this metric (e.g. total instances, total surface area in nm^2)
    total: float
    # Optional: if segment labels should use different values than the bar widths (e.g. for Volume)
    label_fractions: Dict[str, List[float]] | None = None


def _read_excel_data(xls_path: str, cell_volume_um3: float) -> List[MetricRow]:
    """Read volume/instance/surface data from Excel file."""
    # Read the Excel file with the second row as header
    df = pd.read_excel(xls_path, header=1)

    # Get unique organelles from the 'Component Name' column
    organelles = df['Component Name'].dropna().unique().tolist()

    # Define the component order we want in the plot
    component_order_all = ['nuclei', 'mito', 'er', 'lysosome', 'lipidDroplet']
    component_order_instance = ['nuclei', 'mito', 'lysosome', 'lipidDroplet']

    # Initialize data storage for metrics
    metrics = {
        'Volume': {'values': [], 'total': 0},
        'Instance number': {'values': [], 'total': 0},
        'Surface area': {'values': [], 'total': 0}
    }

    # Process Volume + Surface across all 5 organelles, but exclude ER for Instance number.
    for org in component_order_all:
        org_data = df[df['Component Name'] == org]

        # Each organelle should have exactly one row for Area [nm^2] and one row for Volume [nm^3].
        # We use their `Sum` values. Instance number uses the corresponding `Count` (e.g. lysosome=212).
        vol_rows = org_data[org_data['Variable'].astype(str).str.lower().str.contains('volume')]
        area_rows = org_data[
            org_data['Variable'].astype(str).str.lower().str.contains('area')
            & ~org_data['Variable'].astype(str).str.lower().str.contains('bounding')
        ]

        volume_sum = float(vol_rows['Sum'].iloc[0]) if not vol_rows.empty else 0.0
        area_sum = float(area_rows['Sum'].iloc[0]) if not area_rows.empty else 0.0

        # Prefer Count from the volume row; fallback to first available Count.
        if not vol_rows.empty:
            instance_count = float(vol_rows['Count'].iloc[0])
        elif not org_data.empty:
            instance_count = float(org_data['Count'].iloc[0])
        else:
            instance_count = 0.0

        # Store the values
        metrics['Volume']['values'].append(volume_sum)
        metrics['Surface area']['values'].append(area_sum)

        # Update totals
        metrics['Volume']['total'] += volume_sum
        metrics['Surface area']['total'] += area_sum

        # Instance number excludes ER
        if org in component_order_instance:
            metrics['Instance number']['values'].append(instance_count)
            metrics['Instance number']['total'] += instance_count

    # Convert to percentages
    # - Volume: convert organelle volume (nm^3) to % of whole-cell volume (um^3)
    # - Instance number / Surface area: convert to % of the sum across the 5 organelles
    cell_volume_nm3 = float(cell_volume_um3) * NM3_PER_UM3

    if cell_volume_nm3 <= 0:
        raise ValueError(f"cell_volume_um3 must be > 0, got {cell_volume_um3}")

    # Keep raw totals for axis mapping
    total_organelle_volume_nm3 = float(metrics['Volume']['total'])
    total_organelle_volume_um3 = total_organelle_volume_nm3 / NM3_PER_UM3

    # For Volume:
    # - bar widths should be relative to the 5 organelles (sum to 100%)
    # - but in-bar labels should be relative to whole-cell volume
    volume_labels_whole_cell_pct = [v / cell_volume_nm3 * 100 for v in metrics['Volume']['values']]
    if total_organelle_volume_nm3 > 0:
        volume_bar_widths = [v / total_organelle_volume_nm3 * 100 for v in metrics['Volume']['values']]
    else:
        volume_bar_widths = [0.0 for _ in metrics['Volume']['values']]

    # Instance + Surface as % of total
    if metrics['Instance number']['total'] > 0:
        metrics['Instance number']['values'] = [v / metrics['Instance number']['total'] * 100 for v in metrics['Instance number']['values']]
    if metrics['Surface area']['total'] > 0:
        metrics['Surface area']['values'] = [v / metrics['Surface area']['total'] * 100 for v in metrics['Surface area']['values']]

    # Create MetricRow objects
    rows = []
    for name, values in metrics.items():
        if name == 'Volume':
            rows.append(MetricRow(
                name=name,
                x_label="Volume (µm³)",
                fractions={'volume': volume_bar_widths},
                total=total_organelle_volume_um3,  # for x-axis: 100% = total_organelle_volume_um3
                label_fractions={'volume': volume_labels_whole_cell_pct}  # for in-bar labels
            ))
        else:
            rows.append(MetricRow(
                name=name,
                x_label=f"{name} (total)",
                fractions={
                    'instance' if 'Instance' in name else 'surface': values['values']
                },
                total=float(metrics[name]['total'])
            ))

    return rows


def _get_color_map() -> Dict[str, str]:
    # 颜色尽量贴近示例图的观感：紫/绿/橙红/浅灰等
    return {
        "er": "#ff00ff",            # magenta
        "mito": "#ff0000",          # red
        "nuclei": "#ffa500",        # orange
        "lysosome": "#0000ff",      # blue
        "lipiddroplet": "#00ffff",  # cyan
    }


def _plot(ax, row_name: str, x_label: str, organelles: List[str], values: List[float], color_map: Dict[str, str], *, x_tick_formatter=None, segment_label_values: List[float] | None = None):
    y = 0
    left = 0.0

    for i, (org, v) in enumerate(zip(organelles, values)):
        ax.barh(
            y=y,
            width=v,
            left=left,
            height=0.6,
            color=color_map[org],
            edgecolor="none",
            label=org,
        )

        label_v = v if segment_label_values is None else segment_label_values[i]

        # 只标注较大的块，风格参考示例图：白色百分号文字
        # Volume 这一行的 label_v 是“占全细胞体积百分比”，可能会 <3%，但仍希望显示（例如 lysosome/lipid droplet）。
        show_label = (label_v >= 3) if row_name != "Volume" else (label_v > 0)
        if show_label:
            ax.text(
                left + v * 0.03,
                y,
                f"{int(round(label_v))}%",
                va="center",
                ha="left",
                color="white" if org != "lipiddroplet" else "black",
                fontweight="bold" if org == "nuclei" else None,
            )

        left += v

    ax.set_xlim(0, 100)
    ax.set_yticks([0])
    ax.set_yticklabels([row_name])
    ax.set_xlabel(x_label)

    if x_tick_formatter is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x_tick_formatter(x)))

    # 虚线参考线（类似示例图的分隔/网格）
    ax.grid(axis="x", linestyle="--", linewidth=1, alpha=0.35)

    # 去掉多余边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _prettify_legend_labels(org: str) -> str:
    # 图例展示更像论文风格
    mapping = {
        "nuclei": "Nucleus",
        "lysosome": "Lysosome",
        "mito": "Mitochondria",
        "lipiddroplet": "Lipid droplet",
        "er": "ER",
    }
    return mapping.get(org, org)


NM3_PER_UM3 = 1e9


def main():
    parser = argparse.ArgumentParser(description="Plot stacked bars for volume/instance/surface organelle fractions.")
    parser.add_argument(
        "--excel",
        # default="/Volumes/T7/20260117_AIAnalysis/code/volume_surf_ins_src/analysis4_OnePageMultiComponent_Average.xls",
        default="/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/volume_surf_ins_src/analysis4_OnePageMultiComponent_Average.xls",
        help="Path to Excel file with organelle data.",
    )
    parser.add_argument(
        "--out",
        default="/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/figures/statistical_chart1/volume_instance_surface_stacked.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI.",
    )
    parser.add_argument(
        "--cell-volume-um3",
        type=float,
        default=1656.181,
        help="Whole-cell volume in cubic micrometers (um^3). Used to compute Volume as % of whole cell.",
    )
    args = parser.parse_args()

    # Match font/size/weight settings used in plot_whole_boxFigure.py
    mpl.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 0.8,
        }
    )

    organelles = ["nuclei", "mito", "er", "lysosome", "lipiddroplet"]
    color_map = _get_color_map()

    # Read data from Excel instead of using example data
    rows = _read_excel_data(args.excel, cell_volume_um3=args.cell_volume_um3)

    # If no data was read, use example data as fallback
    if not rows or all(len(row.fractions[next(iter(row.fractions))]) == 0 for row in rows):
        print("Warning: Could not read data from Excel. Using example data instead.")
        rows = [
            MetricRow(
                name="Volume",
                x_label="Cell Volume (%)",
                fractions={"volume": [35, 10, 18, 7, 30]},
                total=0,
            ),
            MetricRow(
                name="Instance number",
                x_label="Instance Number (%)",
                fractions={"instance": [28, 16, 22, 10, 24]},
                total=0,
            ),
            MetricRow(
                name="Surface area",
                x_label="Surface Area (%)",
                fractions={"surface": [25, 14, 20, 9, 32]},
                total=0,
            ),
        ]

    fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(14 * 1.5, 5.2), constrained_layout=False)
    if len(rows) == 1:
        axes = [axes]

    organelles_instance = ["nuclei", "mito", "lysosome", "lipiddroplet"]

    for ax, row in zip(axes, rows):
        # 每行只取一个 key（volume/instance/surface），但保持函数通用
        key = next(iter(row.fractions.keys()))
        values = row.fractions[key]

        row_organelles = organelles_instance if row.name == "Instance number" else organelles

        if len(values) != len(row_organelles):
            raise ValueError(f"{row.name}: fractions length {len(values)} != organelles {len(row_organelles)}")

        # 验证总和是否为100%（允许小的浮点误差）
        # 注意：Volume 现在按“占全细胞体积百分比”计算，因此 5 种细胞器的总和不一定是 100%。
        s = float(np.sum(values))
        if row.name != "Volume" and (not np.isclose(s, 100.0, atol=1e-2)):
            print(f"Warning: {row.name} fractions sum to {s:.2f}%, normalizing to 100%")
            values = [v / s * 100 for v in values]

        if row.name == "Volume":
            total_organelle_volume_um3 = float(row.total)

            volume_major_step_um3 = 100.0

            def _fmt_tick(pct: float) -> str:
                # Map 0-100% to 0-total_organelle_volume_um3 (100% -> total of 5 organelles)
                um3 = pct / 100.0 * total_organelle_volume_um3
                return f"{um3:.1e}" if um3 >= 1000 else f"{um3:g}"

            label_values = None
            if row.label_fractions is not None:
                label_values = row.label_fractions.get(key)

            _plot(
                ax,
                row.name,
                "Volume (µm³)",
                row_organelles,
                values,
                color_map,
                x_tick_formatter=_fmt_tick,
                segment_label_values=label_values,
            )

            if total_organelle_volume_um3 > 0:
                ax.xaxis.set_major_locator(MultipleLocator(volume_major_step_um3 / total_organelle_volume_um3 * 100.0))
        elif row.name == "Instance number":
            # 100% corresponds to the total instance count across all 5 organelles
            total_instances = float(row.total)
            instance_major_step = 100.0

            def _fmt_tick(pct: float) -> str:
                n = pct / 100.0 * total_instances
                return f"{int(round(n))}"

            _plot(
                ax,
                row.name,
                "Instance Number (total)",
                row_organelles,
                values,
                color_map,
                x_tick_formatter=_fmt_tick,
            )

            if total_instances > 0:
                ax.xaxis.set_major_locator(MultipleLocator(instance_major_step / total_instances * 100.0))
        elif row.name == "Surface area":
            # 100% corresponds to the total surface area (sum of Area [nm^2] Sum) across all 5 organelles
            total_surface_nm2 = float(row.total)
            total_surface_um2 = total_surface_nm2 / 1e6
            surface_major_step_um2 = 500.0

            def _fmt_tick(pct: float) -> str:
                s_um2 = pct / 100.0 * total_surface_um2
                # Avoid scientific notation; show plain numbers with thousands separators (e.g. 1,000 not 1e3)
                return f"{s_um2:,.0f}"

            _plot(
                ax,
                row.name,
                "Surface Area (µm²)",
                organelles,
                values,
                color_map,
                x_tick_formatter=_fmt_tick,
            )

            if total_surface_um2 > 0:
                ax.xaxis.set_major_locator(MultipleLocator(surface_major_step_um2 / total_surface_um2 * 100.0))
        else:
            _plot(ax, row.name, row.x_label, organelles, values, color_map)

    # 只在顶部做一个总图例（避免每行重复）
    handles = []
    labels = []
    for org in organelles:
        patch = plt.Rectangle((0, 0), 1, 1, color=color_map[org])
        handles.append(patch)
        labels.append(_prettify_legend_labels(org))

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(organelles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    # 给顶部图例留空间，避免与第一行子图重叠
    fig.subplots_adjust(top=0.78, hspace=1.3)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
