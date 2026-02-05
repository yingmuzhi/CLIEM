#!/usr/bin/env python3
"""
High-level drift correction orchestrator.

当前脚本用于在「part 级别」之上做一层封装：
- `--input-dir` 只能是一个目录；
- 如果该目录下只有一个 part（例如 `part1`），
  则对该 part 调用 `drift_part.py` 中与目录处理相同的逻辑；
- 对应 part 的 `--transformations` 自动指向该 part 目录下的 `Results.csv`，
  例如：`.../2_pad/part1/Results.csv`。

后续可以在此基础上继续扩展多 part 的处理逻辑。
"""

import argparse
import os
from types import SimpleNamespace
from typing import List, Dict, Any

import pandas as pd

import drift_part


def parse_arguments():
    """Parse command line arguments for high-level drift correction."""
    parser = argparse.ArgumentParser(
        description="High-level drift correction over part directories (wrapper for drift_part.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 目录下只有一个 part（例如 2_pad/part1），自动用 part1 的 Results.csv 做位移矩阵
  python drift_correction.py -i /path/to/2_pad -o /path/to/2_pad_corrected
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="/Volumes/T7/20251010_halfCell/preprocessedData_Part/2_pad",
        help="输入目录：其下包含一个或多个 part 子目录（如 part1, part2 等）",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/Volumes/T7/20251010_halfCell/preprocessedData_Part/3_drift",
        help="输出目录：对每个 part 在此目录下创建同名子目录保存结果",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*.tif",
        help="匹配 TIF 文件的模式，透传给 drift_part.py（默认：*.tif）",
    )

    parser.add_argument(
        "--bit-depth",
        choices=["8", "16"],
        default="8",
        help="输出位深，透传给 drift_part.py：8 或 16（默认：8）",
    )

    parser.add_argument(
        "--lzw-compression",
        type=bool,
        default=True,
        help="是否使用 LZW 压缩，可显式设置 True/False（默认：True）",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        default=True,
        help="打印详细信息，可显式设置 True/False（默认：True）",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="预处理位移矩阵时使用的阈值，透传给 drift_part.py（默认：50.0）",
    )

    parser.add_argument(
        "--drift-gap",
        type=bool,
        default=True,
        help="是否启用 drift gap 修正，透传给 drift_part.py（默认：True）",
    )

    return parser.parse_args()


def find_part_subdirs(root_dir: str):
    """在 root_dir 下找到所有 part 子目录（简单按目录名以 'part' 开头判断）。"""
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"输入路径不是目录：{root_dir}")

    candidates = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path) and name.lower().startswith("part"):
            candidates.append(full_path)

    return sorted(candidates)


def calculate_gap(part_dirs: List[str], input_root: str) -> str:
    """
    Calculate gap information between consecutive parts based on their Results.csv files.

    Returns the path of the generated gap_results.csv file.
    """
    part_stats: List[Dict[str, Any]] = []

    for part_dir in part_dirs:
        part_name = os.path.basename(os.path.normpath(part_dir))
        results_path = os.path.join(part_dir, "Results.csv")

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"未找到该 part 的 Results.csv：{results_path}")

        df = pd.read_csv(results_path)
        required_cols = {"Slice", "dX", "dY"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{results_path} 缺少必要列：{required_cols}")

        df = df.dropna(subset=["Slice", "dX", "dY"]).copy()
        if df.empty:
            raise ValueError(f"{results_path} 中没有可用的 slice 数据")

        df[["Slice", "dX", "dY"]] = df[["Slice", "dX", "dY"]].apply(pd.to_numeric, errors="raise")

        min_idx = df["Slice"].idxmin()
        max_idx = df["Slice"].idxmax()
        min_row = df.loc[min_idx]
        max_row = df.loc[max_idx]

        part_stats.append(
            {
                "name": part_name,
                "min": {
                    "slice": int(min_row["Slice"]),
                    "dx": float(min_row["dX"]),
                    "dy": float(min_row["dY"]),
                },
                "max": {
                    "slice": int(max_row["Slice"]),
                    "dx": float(max_row["dX"]),
                    "dy": float(max_row["dY"]),
                },
            }
        )

    gap_rows = []
    for idx in range(len(part_stats) - 1):
        current = part_stats[idx]
        nxt = part_stats[idx + 1]

        gap_value = nxt["min"]["slice"] - current["max"]["slice"]
        dx_gap = nxt["min"]["dx"] - current["max"]["dx"]
        dy_gap = nxt["min"]["dy"] - current["max"]["dy"]

        gap_label = f"{current['name']}t{nxt['name']} (Δslice={gap_value})"

        gap_rows.append(
            {
                "index": idx + 1,
                "gap": gap_label,
                "dX": -dx_gap,
                "dY": -dy_gap,
            }
        )

    gap_df = pd.DataFrame(gap_rows, columns=["index", "gap", "dX", "dY"])
    output_path = os.path.join(input_root, "gap_results.csv")
    gap_df.to_csv(output_path, index=False)

    print(f"gap 计算完成，结果已写入：{output_path}")
    return output_path


def modify_results_csv(part_dirs: List[str], gap_df: pd.DataFrame) -> None:
    """
    根据 gap_results DataFrame 对后续 part 的 Results.csv 进行累加偏移。
    """
    required_cols = {"dX", "dY"}
    if not required_cols.issubset(gap_df.columns):
        raise ValueError(f"gap 结果缺少必要列：{required_cols}")

    expected_rows = len(part_dirs) - 1
    if len(gap_df) != expected_rows:
        raise ValueError(
            f"gap 记录数量 ({len(gap_df)}) 与 part 数量不匹配（期望 {expected_rows}）"
        )

    cumulative_offsets = [{"dX": 0.0, "dY": 0.0} for _ in part_dirs]
    running_dx = 0.0
    running_dy = 0.0
    for idx in range(1, len(part_dirs)):
        running_dx += float(gap_df.iloc[idx - 1]["dX"])
        running_dy += float(gap_df.iloc[idx - 1]["dY"])
        cumulative_offsets[idx]["dX"] = running_dx
        cumulative_offsets[idx]["dY"] = running_dy

    for part_dir, offsets in zip(part_dirs, cumulative_offsets):
        if offsets["dX"] == 0 and offsets["dY"] == 0:
            continue

        results_path = os.path.join(part_dir, "Results.csv")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"未找到该 part 的 Results.csv：{results_path}")

        df = pd.read_csv(results_path)
        if not {"dX", "dY"}.issubset(df.columns):
            raise ValueError(f"{results_path} 缺少 dX/dY 列")

        df["dX"] = df["dX"].astype(float) + offsets["dX"]
        df["dY"] = df["dY"].astype(float) + offsets["dY"]

        df.to_csv(results_path, index=False)
        print(
            f"{os.path.basename(part_dir)} 的 Results.csv 已应用偏移："
            f"dX += {offsets['dX']}, dY += {offsets['dY']}"
        )


def process_single_part(part_dir: str, output_root: str, args) -> bool:
    """
    对单个 part 目录调用 drift_part 的「目录处理」逻辑。

    - 输入目录：part_dir（包含若干 TIF 切片）
    - 输出目录：output_root/part_name
    - 位移矩阵：part_dir/Results.csv
    """
    part_name = os.path.basename(os.path.normpath(part_dir))
    part_output_dir = os.path.join(output_root, part_name)

    transformations_path = os.path.join(part_dir, "Results.csv")

    if not os.path.exists(transformations_path):
        raise FileNotFoundError(f"未找到该 part 的 Results.csv：{transformations_path}")

    use_lzw = args.lzw_compression
    verbose = args.verbose
    bit_depth = int(args.bit_depth)

    # 构造一个“精简版”的 args，供 drift_part.process_directory 使用
    inner_args = SimpleNamespace(
        pattern=args.pattern,
        threshold=args.threshold,
        drift_gap=args.drift_gap,
    )

    print(f"=== 处理单个 part ===")
    print(f"part 目录       : {part_dir}")
    print(f"输出目录        : {part_output_dir}")
    print(f"位移矩阵 (CSV)  : {transformations_path}")
    print(f"阈值            : {args.threshold}")
    print(f"输出位深        : {bit_depth}")
    print(f"LZW 压缩        : {'启用' if use_lzw else '关闭'}")
    print()

    return drift_part.process_directory(
        part_dir,
        part_output_dir,
        transformations_path,
        inner_args,
        use_lzw,
        verbose,
        bit_depth,
    )


def main():
    args = parse_arguments()

    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)

    print("=== Drift Correction Orchestrator ===")
    print(f"输入根目录: {input_root}")
    print(f"输出根目录: {output_root}")
    print()

    part_dirs = find_part_subdirs(input_root)

    if not part_dirs:
        raise RuntimeError(f"在目录下未找到任何 part 子目录：{input_root}")

    if len(part_dirs) == 1:
        # 你现在描述的「只有一个 part」场景：
        # 直接对该 part 调用与 drift_part.py 目录模式相同的逻辑，
        # 且 transformations 自动使用 part 下的 Results.csv。
        success = process_single_part(part_dirs[0], output_root, args)
    else:
        gap_csv_path = calculate_gap(part_dirs, input_root)
        gap_df = pd.read_csv(gap_csv_path)
        modify_results_csv(part_dirs, gap_df)
        success = True
        for part_dir in part_dirs:
            part_success = process_single_part(part_dir, output_root, args)
            success = success and part_success

    if success:
        print("=== Drift correction completed successfully! ===")
        return 0
    else:
        print("=== Drift correction completed with errors! ===")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())