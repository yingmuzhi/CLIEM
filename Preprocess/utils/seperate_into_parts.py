import os
import shutil
from pathlib import Path


def split_folder_images(
    src_root: Path,
    dst_root: Path,
    part_configs,
) -> None:
    """
    将 src_root 下的每个一级子文件夹中的图像文件按给定范围划分到 part1-4，
    并在对应 part 目录中拷贝指定的 Results.csv。

    part_configs: 列表，元素为字典：
        {
            "name": "part1",
            "start": 1,
            "end": 88,
            "results_csv": Path(...)
        }
    """
    if not src_root.exists():
        raise FileNotFoundError(f"源目录不存在: {src_root}")

    # 允许的图像后缀（全部转为小写比较）
    image_exts = {".tif", ".tiff", ".tifff", ".png", ".jpg", ".jpeg"}

    for sub in sorted(p for p in src_root.iterdir() if p.is_dir()):
        # 例如 sub = /.../output/autosome
        rel_name = sub.name
        dst_sub_root = dst_root / rel_name

        print(f"\n处理子目录: {sub}")

        # 收集该子目录下的所有图像文件（排除目录），并按名称排序
        # 这里显式按后缀过滤，确保 .tif/.tiff 也会被包含
        all_files = sorted(
            f
            for f in sub.iterdir()
            if f.is_file()
            and f.suffix.lower() in image_exts
            and not f.name.startswith("._")
        )

        if not all_files:
            print(f"  警告: 目录下没有文件，跳过: {sub}")
            continue

        total = len(all_files)
        print(f"  共找到 {total} 个文件")

        # 逐个 part 处理
        for cfg in part_configs:
            part_name = cfg["name"]
            start_idx = cfg["start"]
            end_idx = cfg["end"]
            results_csv: Path = cfg["results_csv"]

            # 目标目录：.../2_seperate/autosome/part1
            dst_part_dir = dst_sub_root / part_name
            dst_part_dir.mkdir(parents=True, exist_ok=True)

            # 索引范围是 1-based，Python 切片是 0-based，且切片右开区间
            # 所以 [start_idx-1 : end_idx]
            start0 = max(start_idx - 1, 0)
            end0 = min(end_idx, total)

            if start0 >= total:
                print(f"  {part_name}: 起始索引 {start_idx} 超出文件数 {total}，本 part 不拷贝图像")
                files_slice = []
            else:
                files_slice = all_files[start0:end0]

            print(
                f"  {part_name}: 拷贝图像索引 {start_idx}-{end_idx} "
                f"(实际 {start0+1}-{start0+len(files_slice)})，数量 {len(files_slice)}"
            )

            # 复制图像文件
            for f in files_slice:
                dst_file = dst_part_dir / f.name
                shutil.copy2(f, dst_file)

            # 复制对应的 Results.csv
            if results_csv.exists():
                dst_results = dst_part_dir / "Results.csv"
                shutil.copy2(results_csv, dst_results)
                print(f"    已拷贝 Results.csv -> {dst_results}")
            else:
                print(f"    警告: 找不到 Results.csv: {results_csv}")


def main():
    # 源输出目录
    src_root = Path("/Volumes/T7/20251010_halfCell/20251119_amira_o/output")

    # 目标目录：与示例中一致，放在 2_seperate 下
    dst_root = Path("/Volumes/T7/20251010_halfCell/20251119_amira_o/2_seperate")

    # Results.csv 来源目录（按题意固定）
    base_results = Path(
        "/Volumes/T7/20251010_halfCell/preprocessedData_Part/1_align"
    )

    part_configs = [
        {
            "name": "part1",
            "start": 1,
            "end": 88,
            "results_csv": base_results / "part1" / "Results.csv",
        },
        {
            "name": "part2",
            "start": 89,
            "end": 143,
            "results_csv": base_results / "part2" / "Results.csv",
        },
        {
            "name": "part3",
            "start": 144,
            "end": 303,
            "results_csv": base_results / "part3" / "Results.csv",
        },
        {
            "name": "part4",
            "start": 304,
            "end": 425,
            "results_csv": base_results / "part4" / "Results.csv",
        },
    ]

    print(f"源目录: {src_root}")
    print(f"目标目录: {dst_root}")
    split_folder_images(src_root, dst_root, part_configs)
    print("\n全部处理完成。")


if __name__ == "__main__":
    main()


