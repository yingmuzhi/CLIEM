"""
Compute instance-segmentation evaluation metrics for organelle predictions.

This script compares instance-labeled prediction TIFFs against instance-labeled
ground-truth TIFFs and writes a JSON report.

Main outputs:
- Per-class TP / FP / FN, Precision / Recall / F1
- Overall micro-averaged TP / FP / FN, Precision / Recall / F1
- Boundary F1 (per class and overall) with configurable pixel tolerance

Expected usage in pipeline:
1) `convert.py` converts prediction masks to instance-label TIFFs.
2) This script (`metric_compute.py`) computes metrics from prediction instances
   and GT instance labels.
3) `metric_plot.py` visualizes the generated metrics JSON.

Notes:
- Input labels use 0 as background and 1..N as instance IDs.
- Class files are discovered by filename patterns in `pred_dir` and `gt_dir`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import tifffile
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'tifffile'. Please install it in your conda env (env_cp311_ymz)."
    ) from e

# For boundary extraction and tolerance matching
try:
    from scipy.ndimage import binary_dilation, binary_erosion
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'scipy'. Please install it in your conda env (env_cp311_ymz)."
    ) from e


def _to_2d_int_labels(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[0]
    return arr.astype(np.int64, copy=False)


def _instances_from_label(label: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.unique(label)
    ids = ids[ids != 0]
    return {int(i): (label == i) for i in ids}


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union else 0.0


def _match_instances_greedy(
    pred_masks: dict[int, np.ndarray],
    gt_masks: dict[int, np.ndarray],
    iou_th: float,
) -> tuple[int, int, int]:
    pred_ids = list(pred_masks.keys())
    gt_ids = list(gt_masks.keys())

    all_pairs: list[tuple[float, int, int]] = []
    for pid in pred_ids:
        pm = pred_masks[pid]
        for gid in gt_ids:
            val = _iou(pm, gt_masks[gid])
            if val >= iou_th:
                all_pairs.append((val, pid, gid))

    all_pairs.sort(reverse=True, key=lambda x: x[0])

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    tp = 0
    for val, pid, gid in all_pairs:
        if pid in used_pred or gid in used_gt:
            continue
        used_pred.add(pid)
        used_gt.add(gid)
        tp += 1

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)


def _boundary_map(binary_mask: np.ndarray) -> np.ndarray:
    """1-pixel boundary of a binary mask."""
    binary_mask = binary_mask.astype(bool)
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    er = binary_erosion(binary_mask)
    return np.logical_and(binary_mask, np.logical_not(er))


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _boundary_f1(
    pred_label: np.ndarray,
    gt_label: np.ndarray,
    tolerance_px: int = 2,
) -> dict:
    """Boundary F1 for a class (instance label images).

    We compute boundaries on the union-of-instances binary mask.

    Matching uses a tolerance (in pixels):
    - A predicted boundary pixel is a TP if it falls within tol of any GT boundary pixel.
    - A GT boundary pixel is a TP if it falls within tol of any pred boundary pixel.

    Returns precision/recall/f1 and raw counts.
    """
    pred_bin = pred_label != 0
    gt_bin = gt_label != 0

    pred_b = _boundary_map(pred_bin)
    gt_b = _boundary_map(gt_bin)

    struct = _disk_structure(int(tolerance_px))

    gt_b_dil = binary_dilation(gt_b, structure=struct)
    pred_b_dil = binary_dilation(pred_b, structure=struct)

    # precision: how many pred boundary pixels match GT boundary within tol
    tp_p = int(np.logical_and(pred_b, gt_b_dil).sum())
    fp_p = int(np.logical_and(pred_b, np.logical_not(gt_b_dil)).sum())

    # recall: how many GT boundary pixels are matched by pred boundary within tol
    tp_r = int(np.logical_and(gt_b, pred_b_dil).sum())
    fn_r = int(np.logical_and(gt_b, np.logical_not(pred_b_dil)).sum())

    precision = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
    recall = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tolerance_px": int(tolerance_px),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_boundary_pixels": int(pred_b.sum()),
        "gt_boundary_pixels": int(gt_b.sum()),
        "tp_pred": tp_p,
        "fp_pred": fp_p,
        "tp_gt": tp_r,
        "fn_gt": fn_r,
    }


def evaluate_instance_segmentation(
    pred_dir: str | Path,
    gt_dir: str | Path,
    iou_threshold: float = 0.5,
    strict_shape: bool = True,
    boundary_tolerance_px: int = 2,
) -> dict:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    # Dynamically find files instead of hardcoding slice names
    class_map = {
        "er": ("er_*.tiff", "*_er.ome.tiff"),
        "lipidDroplet": ("lipidDroplet_*.tiff", "*_lipidDroplet.ome.tiff"),
        "lysosome": ("lysosome_*.tiff", "*_lysosome.ome.tiff"),
        "mito": ("mito_*.tiff", "*_mitocondria.ome.tiff"),
        "nuclei": ("nuclei_*.tiff", "*_nuclei.ome.tiff"),
    }

    def _find_unique_file(directory: Path, pattern: str) -> Path | None:
        files = list(directory.glob(pattern))
        if not files:
            return None
        if len(files) > 1:
            print(f"Warning: Multiple files matching '{pattern}' in {directory}, using first one: {files[0]}")
        return files[0]

    pairs = []
    for cls, (pred_pat, gt_pat) in class_map.items():
        pred_path = _find_unique_file(pred_dir, pred_pat)
        gt_path = _find_unique_file(gt_dir, gt_pat)
        
        if pred_path is None:
            print(f"Warning: Skipping class '{cls}' - no prediction file matching '{pred_pat}' in {pred_dir}")
            continue
        if gt_path is None:
            print(f"Warning: Skipping class '{cls}' - no ground truth file matching '{gt_pat}' in {gt_dir}")
            continue
            
        pairs.append((cls, pred_path, gt_path))
    
    if not pairs:
        raise FileNotFoundError(
            f"No valid class files found in prediction dir: {pred_dir} and GT dir: {gt_dir}"
        )
    
    print(f"Found {len(pairs)} classes to evaluate:")
    for cls, pred_path, gt_path in pairs:
        print(f"  {cls}: {pred_path.name} vs {gt_path.name}")

    per_class: dict[str, dict] = {}
    TP = FP = FN = 0

    # For overall boundary f1 (micro over boundary pixels): we aggregate TP/FP/FN counts.
    bf1_tp_pred = bf1_fp_pred = bf1_tp_gt = bf1_fn_gt = 0

    for cls, pred_path, gt_path in pairs:
        pred = _to_2d_int_labels(tifffile.imread(str(pred_path)))
        gt = _to_2d_int_labels(tifffile.imread(str(gt_path)))

        if strict_shape and pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for class '{cls}': pred{pred.shape} vs gt{gt.shape}"
            )

        pred_masks = _instances_from_label(pred)
        gt_masks = _instances_from_label(gt)

        tp, fp, fn = _match_instances_greedy(pred_masks, gt_masks, iou_threshold)
        TP += tp
        FP += fp
        FN += fn

        p, r, f1 = _prf(tp, fp, fn)

        bf1 = _boundary_f1(pred, gt, tolerance_px=boundary_tolerance_px)
        bf1_tp_pred += int(bf1["tp_pred"])
        bf1_fp_pred += int(bf1["fp_pred"])
        bf1_tp_gt += int(bf1["tp_gt"])
        bf1_fn_gt += int(bf1["fn_gt"])

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "n_pred": len(pred_masks),
            "n_gt": len(gt_masks),
            "boundary_f1": bf1,
            "pred_file": str(pred_path),
            "gt_file": str(gt_path),
        }

    P, R, F1 = _prf(TP, FP, FN)

    # overall boundary f1 (micro)
    bf1_precision = bf1_tp_pred / (bf1_tp_pred + bf1_fp_pred) if (bf1_tp_pred + bf1_fp_pred) else 0.0
    bf1_recall = bf1_tp_gt / (bf1_tp_gt + bf1_fn_gt) if (bf1_tp_gt + bf1_fn_gt) else 0.0
    bf1_f1 = (2 * bf1_precision * bf1_recall / (bf1_precision + bf1_recall)) if (bf1_precision + bf1_recall) else 0.0

    result = {
        "iou_threshold": float(iou_threshold),
        "boundary_f1": {
            "tolerance_px": int(boundary_tolerance_px),
            "precision": float(bf1_precision),
            "recall": float(bf1_recall),
            "f1": float(bf1_f1),
            "tp_pred": int(bf1_tp_pred),
            "fp_pred": int(bf1_fp_pred),
            "tp_gt": int(bf1_tp_gt),
            "fn_gt": int(bf1_fn_gt),
        },
        "overall_micro": {
            "tp": TP,
            "fp": FP,
            "fn": FN,
            "precision": P,
            "recall": R,
            "f1": F1,
        },
        "per_class": per_class,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
    }

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/prediction_instances/320",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/ground_truth/320",
    )
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument(
        "--boundary_tol",
        type=int,
        default=6,
        help="Boundary F1 tolerance in pixels.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/Volumes/T7/20251204_halfCell/CLIEM_code/Analysis/src/toy_example/metrics_320_iou0.3.json",
    )

    args = parser.parse_args()

    res = evaluate_instance_segmentation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        iou_threshold=args.iou,
        boundary_tolerance_px=args.boundary_tol,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote", str(out_path))
    print("overall_micro", res["overall_micro"])
    print("boundary_f1  ", res["boundary_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
