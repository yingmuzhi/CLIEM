#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

PART1_DIR = Path("/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1")
MERGE_DIR = Path("/Volumes/DATA5/src/4_merge")
DEST_DIR = Path("/Volumes/DATA5/src/4_merge_new")
DEST_TEMPLATE = "slice_{:06d}.tif"


def natural_key(path: Path) -> Tuple:
    """用于自然排序的键，确保数字顺序正确。"""
    import re

    parts = re.split(r"(\d+)", path.name)
    key: List[Tuple[int, str]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def list_tifs(directory: Path) -> List[Path]:
    """列出目录下所有非隐藏 tif 文件，按自然顺序排序。"""
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    candidates = [
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".tif"
    ]
    return sorted(candidates, key=natural_key)


def ensure_dest_dir(dest: Path) -> None:
    """确保目标目录存在。"""
    dest.mkdir(parents=True, exist_ok=True)


def main() -> None:
    part1_files = list_tifs(PART1_DIR)
    merge_files = list_tifs(MERGE_DIR)

    ordered_sources = part1_files + merge_files

    if not ordered_sources:
        raise RuntimeError("未找到任何可处理的 tif 文件。")

    ensure_dest_dir(DEST_DIR)

    for idx, src in enumerate(ordered_sources):
        dest_name = DEST_TEMPLATE.format(idx)
        dest_path = DEST_DIR / dest_name
        shutil.copy2(src, dest_path)
        print(f"[COPY] {src} -> {dest_path}")

    print(f"[DONE] 复制完成，共处理 {len(ordered_sources)} 个文件。")


if __name__ == "__main__":
    main()
