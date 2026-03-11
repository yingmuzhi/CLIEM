from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = sys.executable
DEFAULT_BASE_DIR = "/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example"

DEFAULT_SLICE = "320"  # can be changed to "244" etc.
DEFAULT_IOU = 0.3

DEFAULT_CONNECTIVITY = 1
DEFAULT_THRESHOLD = 0.0


def _paths_for_slice(base_dir: str, slice_id: str, iou: float) -> dict[str, str]:
    base = Path(base_dir)
    return {
        "pred_dir": str(base / "prediction" / str(slice_id)),
        "pred_instances_dir": str(base / "prediction_instances" / str(slice_id)),
        "gt_dir": str(base / "ground_truth" / str(slice_id)),
        "metrics_json": str(base / f"metrics_{slice_id}_iou{iou}.json"),
        "plot_out_dir": str(base / "figures" / f"confusion_plots_{slice_id}_iou{iou}"),
    }


def _run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _has_valid_tiffs(directory: str) -> bool:
    path = Path(directory)
    if not path.is_dir():
        return False
    patterns = ("*.tiff", "*.tif", "*.ome.tiff")
    files: list[Path] = []
    for pat in patterns:
        files.extend(path.glob(pat))
    files = [fp for fp in files if not fp.name.startswith(".")]
    return bool(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full analysis pipeline: convert prediction -> compute metrics -> plot confusion matrices."
        )
    )

    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--slice",
        type=str,
        default=DEFAULT_SLICE,
        help="Slice folder name, e.g. 320 / 244 (used to build default paths)",
    )

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)

    parser.add_argument("--connectivity", type=int, default=DEFAULT_CONNECTIVITY, choices=[1, 2])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--title_prefix", type=str, default="")

    # Allow overriding any derived paths
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--pred_instances_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--metrics_json", type=str, default=None)
    parser.add_argument("--plot_out_dir", type=str, default=None)

    args = parser.parse_args()

    derived = _paths_for_slice(args.base_dir, args.slice, args.iou)
    pred_dir = args.pred_dir or derived["pred_dir"]
    pred_instances_dir = args.pred_instances_dir or derived["pred_instances_dir"]
    gt_dir = args.gt_dir or derived["gt_dir"]
    metrics_json = args.metrics_json or derived["metrics_json"]
    plot_out_dir = args.plot_out_dir or derived["plot_out_dir"]

    if not _has_valid_tiffs(pred_dir):
        raise SystemExit(f"No valid TIFF files found in pred_dir: {pred_dir}")
    if not _has_valid_tiffs(gt_dir):
        raise SystemExit(f"No valid TIFF files found in gt_dir: {gt_dir}")

    base = Path(args.base_dir)
    convert_py = str(PROJECT_ROOT / "utils" / "convert.py")
    metric_compute_py = str(PROJECT_ROOT / "utils" / "metric_compute.py")
    metric_plot_py = str(PROJECT_ROOT / "utils" / "metric_plot.py")

    # 1) Convert prediction masks to instance-label masks
    _run(
        [
            args.python,
            convert_py,
            "--input_dir",
            pred_dir,
            "--output_dir",
            pred_instances_dir,
            "--connectivity",
            str(args.connectivity),
            "--threshold",
            str(args.threshold),
        ]
    )

    # 2) Compute metrics JSON
    _run(
        [
            args.python,
            metric_compute_py,
            "--pred_dir",
            pred_instances_dir,
            "--gt_dir",
            gt_dir,
            "--iou",
            str(args.iou),
            "--out",
            metrics_json,
        ]
    )

    # 3) Plot confusion matrices
    _run(
        [
            args.python,
            metric_plot_py,
            "--metrics_json",
            metrics_json,
            "--out_dir",
            plot_out_dir,
            "--title_prefix",
            args.title_prefix,
        ]
    )

    print("\nPipeline complete.")
    print("slice      :", args.slice)
    print("metrics_json:", metrics_json)
    print("plots_dir  :", plot_out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
