#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from dataclasses import dataclass
from typing import Dict, List, Optional

# 常量定义
# 注意：contactSites_src 的 analysis6_OnePageMultiComponent_Average.xls 中 Area 的原始单位为 µm²
# 因此这里不做 nm²→µm² 的换算。

@dataclass
class PlotData:
    """存储绘图数据的类"""
    name: str
    x_label: str
    values: List[float]
    total: float
    component_names: List[str]
    label_values: Optional[List[float]] = None

def read_contact_sites_data(filepath: str) -> Dict[str, PlotData]:
    """Read contact sites data from Excel file."""
    try:
        df = pd.read_excel(filepath, header=1)
        components = {
            'lysosome Selection_ERLess30nm': 'Lysosome-ER Contact',
            'mito Selection_ERLess30nm': 'Mitocondria-ER Contact'
        }
        
        instance_values, area_values = [], []
        instance_labels, area_labels = [], []

        for comp_id, display_name in components.items():
            comp_data = df[df['Component Name'] == comp_id]
            if comp_data.empty:
                print(f"Warning: No data found for component {comp_id}")
                instance_values.append(0)
                area_values.append(0)
                instance_labels.append(0)
                area_labels.append(0)
                continue

            count_row = comp_data[comp_data['Variable'].str.contains('Volume', na=False, case=False)]
            count = float(count_row['Count'].iloc[0]) if not count_row.empty and 'Count' in count_row.columns else 0
            if count == 0:
                 for _, row in comp_data.iterrows():
                    if 'Count' in row and not pd.isna(row['Count']):
                        count = float(row['Count'])
                        break
            instance_values.append(count)
            instance_labels.append(count)

            area_row = comp_data[comp_data['Variable'].str.contains('Area', na=False, case=False) & ~comp_data['Variable'].str.contains('Bounding', na=False, case=False)]
            area = float(area_row['Sum'].iloc[0]) if not area_row.empty and 'Sum' in area_row.columns else 0
            area_values.append(area)
            area_labels.append(area)

        total_instance = sum(instance_values)
        total_area = sum(area_values)

        instance_percent = [v/total_instance*100 if total_instance > 0 else 0 for v in instance_values]
        area_percent = [v/total_area*100 if total_area > 0 else 0 for v in area_values]

        return {
            'Instance number': PlotData(
                name='Instance number',
                x_label='Instance Number (total)',
                values=instance_percent,
                total=total_instance,
                component_names=list(components.values()),
                label_values=instance_labels
            ),
            'Surface area': PlotData(
                name='Surface area',
                x_label='Surface Area (µm²)',
                values=area_percent,
                total=total_area,
                component_names=list(components.values()),
                label_values=area_labels
            )
        }
    except Exception as e:
        print(f"Error reading data: {str(e)}")
        raise

def plot_combined_metrics(data_dict: Dict[str, PlotData], output_file: str, dpi: int = 300):
    """将多个指标绘制在同一个Figure中"""

    # 统一字体大小（图例/坐标轴/刻度/条内标注全部使用同一字号）
    FONT_SIZE = 24

    mpl.rcParams.update({
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'axes.linewidth': 0.8,
    })

    fig, axes = plt.subplots(nrows=len(data_dict), ncols=1, figsize=(14, 5.2), constrained_layout=False)
    if len(data_dict) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e']

    for ax, data in zip(axes, data_dict.values()):
        left = 0
        for i, (value, component) in enumerate(zip(data.values, data.component_names)):
            if value <= 0: continue
            ax.barh(0, value, left=left, height=0.6, color=colors[i % len(colors)], edgecolor='none', label=component)
            label_value = data.label_values[i] if data.label_values else value
            label_text = f"{value:.1f}%\n({int(round(label_value))})" if 'Instance' in data.name else f"{value:.1f}%\n({label_value:.2f} µm²)"
            if value > 5:
                ax.text(left + value * 0.5, 0, label_text, va='center', ha='center', color='white', fontweight='bold', fontsize=FONT_SIZE)
            left += value

        ax.set_xlim(0, 100)
        ax.set_yticks([0])
        ax.set_yticklabels([data.name])
        ax.set_xlabel(data.x_label)
        ax.grid(axis='x', linestyle='--', linewidth=1, alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)

        if data.name == "Instance number":
            total_instances = float(data.total)
            major_step = 20.0

            # 仿照 plot_volume_surface_instance.py：横轴仍是 0-100(%)，但 tick label 映射为真实总数
            pct_ticks = np.arange(0.0, 100.0 + 1e-9, (major_step / total_instances * 100.0) if total_instances > 0 else 100.0)
            pct_ticks = np.clip(pct_ticks, 0, 100)
            pct_ticks = np.unique(np.round(pct_ticks, 8))
            if len(pct_ticks) == 0 or pct_ticks[0] != 0:
                pct_ticks = np.insert(pct_ticks, 0, 0.0)
            if pct_ticks[-1] != 100.0:
                pct_ticks = np.append(pct_ticks, 100.0)

            tick_labels = [f"{int(round(p / 100.0 * total_instances))}" for p in pct_ticks]
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(pct_ticks.tolist()))
            ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(tick_labels))

        elif data.name == "Surface area":
            total_surface_um2 = float(data.total)
            major_step_um2 = 90.0

            # 同上：tick label 映射为真实总面积(µm²)，最小刻度 90 µm²
            pct_ticks = np.arange(0.0, 100.0 + 1e-9, (major_step_um2 / total_surface_um2 * 100.0) if total_surface_um2 > 0 else 100.0)
            pct_ticks = np.clip(pct_ticks, 0, 100)
            pct_ticks = np.unique(np.round(pct_ticks, 8))
            if len(pct_ticks) == 0 or pct_ticks[0] != 0:
                pct_ticks = np.insert(pct_ticks, 0, 0.0)
            if pct_ticks[-1] != 100.0:
                pct_ticks = np.append(pct_ticks, 100.0)

            def _format_area_tick(v_um2: float) -> str:
                if v_um2 == 0:
                    return "0"
                if v_um2 < 1:
                    return f"{v_um2:.2f}"
                if v_um2 < 1000:
                    return f"{v_um2:.0f}"
                return f"{v_um2:.1e}"

            tick_labels = [_format_area_tick(p / 100.0 * total_surface_um2) for p in pct_ticks]
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(pct_ticks.tolist()))
            ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(tick_labels))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.12), handlelength=1.6, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(top=0.78, hspace=1.3)
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot contact sites data.')
    parser.add_argument('--input', type=str, default='/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/contactSites_src/analysis6_OnePageMultiComponent_Average.xls', help='Path to input Excel file')
    parser.add_argument('--output', type=str, default='/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/figures/statistical_chart2/contact_sites_combined.png', help='Output image file path')
    parser.add_argument('--dpi', type=int, default=300, help='Output image DPI (default: 300)')
    args = parser.parse_args()

    try:
        data_dict = read_contact_sites_data(args.input)
        plot_combined_metrics(data_dict, args.output, args.dpi)
        print("Plot generation successful!")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())