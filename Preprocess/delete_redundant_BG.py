#!/usr/bin/env python3
"""
Utilities for trimming redundant zero-background areas from 2D slice stacks.

This module exposes a single entry-point, `delete_redundant_background`, which
scans all slices under a directory, finds the tightest bounding box that
contains non-zero pixels across the stack (the same logic used previously
inside rotate_sample.py), and writes the cropped slices to a target directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile

__all__ = ["delete_redundant_background"]


@dataclass(frozen=True)
class CropWindow:
    """Represents the y/x slice to apply to each 2D image."""

    y_slice: slice
    x_slice: slice

    @property
    def size(self) -> tuple[int, int]:
        return (
            self.y_slice.stop - self.y_slice.start,
            self.x_slice.stop - self.x_slice.start,
        )


def _gather_slice_paths(stack_dir: Path, pattern: str) -> list[Path]:
    stack_dir = Path(stack_dir)
    slice_paths = sorted(
        p
        for p in stack_dir.glob(pattern)
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in {".tif", ".tiff"}
    )
    return slice_paths


def _compute_crop_window(slice_paths: Sequence[Path]) -> tuple[CropWindow | None, tuple[int, int] | None]:
    mask_xy = None
    reference_shape = None

    for path in slice_paths:
        slice_arr = tifffile.imread(str(path))
        if slice_arr.ndim != 2:
            raise ValueError(
                f"Expected 2D slice but got shape {slice_arr.shape} for {path.name}"
            )

        if reference_shape is None:
            reference_shape = slice_arr.shape
            mask_xy = np.zeros(reference_shape, dtype=bool)
        elif slice_arr.shape != reference_shape:
            raise ValueError(
                "Inconsistent slice sizes detected while computing delete_background "
                f"bounds: first slice {reference_shape}, current slice {slice_arr.shape}"
            )

        mask_xy |= slice_arr != 0

    if mask_xy is None:
        return None, reference_shape

    if not mask_xy.any():
        # All pixels are zero; keep the original size but still produce a window
        y_slice = slice(0, reference_shape[0])
        x_slice = slice(0, reference_shape[1])
        return CropWindow(y_slice, x_slice), reference_shape

    y_mask = mask_xy.any(axis=1)
    x_mask = mask_xy.any(axis=0)
    y_indices = np.where(y_mask)[0]
    x_indices = np.where(x_mask)[0]

    y_slice = slice(y_indices[0], y_indices[-1] + 1)
    x_slice = slice(x_indices[0], x_indices[-1] + 1)
    return CropWindow(y_slice, x_slice), reference_shape


def _write_cropped_slices(
    slice_paths: Iterable[Path],
    crop_window: CropWindow,
    output_dir: Path,
    compression: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_paths = list(slice_paths)
    total = len(slice_paths)

    for idx, path in enumerate(slice_paths, start=1):
        slice_arr = tifffile.imread(str(path))
        cropped = slice_arr[crop_window.y_slice, crop_window.x_slice]
        out_path = output_dir / path.name
        tifffile.imwrite(str(out_path), cropped, compression=compression)
        if idx % 100 == 0 or idx == total:
            print(f"  Saved {idx}/{total} slices", flush=True)


def delete_redundant_background(
    stack_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.tif",
    use_lzw: bool = True,
) -> None:
    """
    Remove redundant zero background by cropping all slices in a stack.

    Parameters
    ----------
    stack_dir:
        Directory containing the flipped slices (input).
    output_dir:
        Directory where cropped slices will be written. Existing files may be
        overwritten if they share the same name.
    pattern:
        Glob pattern used to discover slice files. Defaults to '*.tif'.
    use_lzw:
        Whether to save with LZW compression (matches flip stage default).
    """

    stack_dir = Path(stack_dir)
    output_dir = Path(output_dir)
    print("\n=== Running delete_background on flipped stack ===")
    print(f"  Input stack:  {stack_dir}")
    print(f"  Output dir:   {output_dir}")

    slice_paths = _gather_slice_paths(stack_dir, pattern)
    if not slice_paths:
        print(
            f"delete_background skipped: no slices matching '{pattern}' found in "
            f"{stack_dir}"
        )
        return

    crop_window, reference_shape = _compute_crop_window(slice_paths)
    if reference_shape is None:
        print("delete_background skipped: unable to determine slice shape.")
        return
    if crop_window is None:
        print(
            "delete_background skipped: unable to determine crop bounds "
            "(no slices loaded)."
        )
        return

    new_height, new_width = crop_window.size
    if (new_height, new_width) == reference_shape:
        print(
            "delete_background: stack already tight or empty; copying slices without "
            "cropping."
        )
    else:
        print(
            f"delete_background: cropping slices from {reference_shape} to "
            f"({new_height}, {new_width})."
        )

    compression = "lzw" if use_lzw else None
    _write_cropped_slices(slice_paths, crop_window, output_dir, compression)

