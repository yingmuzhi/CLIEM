"""Convert prediction masks to instance-label TIFF files.

This script reads semantic/binary prediction masks from an input directory,
extracts connected components from each file, and writes instance-ID label maps
to an output directory.

Main behavior:
- Read TIFF prediction masks and squeeze to 2D when needed.
- Binarize by threshold (foreground: value > threshold).
- Run connected-component labeling (4- or 8-connectivity).
- Save results as int32 TIFF, with labels: 0=background, 1..N=instances.
- Ignore hidden/system files whose names start with ".".

Typical use in this project:
- Called as step 1 by analysis_pipeline*.py.
- Output is consumed by metric_compute.py for instance-level evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'tifffile'. Please install it in your conda env (env_cp311_ymz)."
    ) from e

# Connected components
try:
    from scipy.ndimage import label as cc_label
except Exception:  # pragma: no cover
    cc_label = None

try:
    from skimage.measure import label as sk_label
except Exception:  # pragma: no cover
    sk_label = None


DEFAULT_INPUT_DIR = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/prediction/320"
DEFAULT_OUTPUT_DIR = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/prediction_instances/320"


def _to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[0]
    return arr


def _connected_components(binary: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Label connected components in a binary mask.

    Returns int32 label image with 0 background and 1..N components.

    connectivity:
      - 1: 4-connectivity
      - 2: 8-connectivity
    """
    binary = (binary != 0)

    if cc_label is not None:
        if connectivity == 1:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        else:
            structure = np.ones((3, 3), dtype=bool)
        lab, _ = cc_label(binary, structure=structure)
        return lab.astype(np.int32, copy=False)

    if sk_label is not None:
        return sk_label(binary.astype(np.uint8), connectivity=connectivity).astype(
            np.int32, copy=False
        )

    raise SystemExit(
        "Need connected-component labeling. Please install scipy or scikit-image in env_cp311_ymz."
    )


def convert_prediction_to_instances(
    input_path: str | Path,
    output_path: str | Path,
    connectivity: int = 1,
    foreground_threshold: float = 0.0,
) -> dict:
    """Convert a prediction mask to instance-ID labels.

    - Reads TIFF
    - Converts to 2D
    - Binarizes: arr > foreground_threshold
    - Connected components => instance labels (0,1..N)
    - Saves as int32 TIFF
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = tifffile.imread(str(input_path))
    arr = _to_2d(arr)

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr)

    binary = arr > foreground_threshold
    lab = _connected_components(binary, connectivity=connectivity)

    tifffile.imwrite(str(output_path), lab.astype(np.int32, copy=False))

    u = np.unique(lab)
    n_inst = int((u != 0).sum())

    return {
        "status": "success",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "in_dtype": str(arr.dtype),
        "out_dtype": str(lab.dtype),
        "out_instances": n_inst,
        "shape": tuple(lab.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert prediction masks to instance-label TIFFs by connected components."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing prediction TIFFs (binary/semantic masks).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write instance-label TIFFs.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=4-connectivity, 2=8-connectivity",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Foreground threshold: pixels > threshold are treated as foreground.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"Error: input_dir does not exist: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Converting prediction masks from '{input_dir}' to instance labels in '{output_dir}'..."
    )

    file_patterns = ["*.tiff", "*.tif", "*.ome.tiff"]
    files = []
    for pat in file_patterns:
        files.extend(input_dir.glob(pat))

    files = sorted(set(files))
    files = [fp for fp in files if not fp.name.startswith(".")]
    if not files:
        print("No TIFF files found.")
        return 0

    ok = 0
    for fp in files:
        out_fp = output_dir / fp.name
        try:
            res = convert_prediction_to_instances(
                input_path=fp,
                output_path=out_fp,
                connectivity=args.connectivity,
                foreground_threshold=args.threshold,
            )
            ok += 1
            print(
                f"  {fp.name}: instances={res['out_instances']} -> {out_fp}"
            )
        except Exception as e:
            print(f"  FAILED {fp.name}: {e}")

    print(f"Done. Converted {ok}/{len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
