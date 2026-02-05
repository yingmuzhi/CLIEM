"""
按照 yz 平面旋转 3D 体数据并切片保存的脚本。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import rotate as nd_rotate
import tifffile


def read_volume(input_dir: Path) -> np.ndarray:
    """将按序排列的 tiff 切片堆叠成体数据。"""
    slice_paths = sorted(
        p
        for p in input_dir.iterdir()
        if not p.name.startswith(".") and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not slice_paths:
        raise FileNotFoundError(f"No tif/tiff slices found in {input_dir}")

    slices = [tifffile.imread(str(p)) for p in slice_paths]
    volume = np.stack(slices, axis=0)
    return volume


def rotate_volume(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """绕 x 轴（即在 yz 平面内）对体数据进行 3D 旋转。"""
    volume_f32 = volume.astype(np.float32, copy=False)
    rotated = nd_rotate(
        volume_f32,
        angle=angle_deg,
        axes=(0, 1),  # rotate within the yz plane → around x axis
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return rotated


def delete_background(volume: np.ndarray) -> np.ndarray:
    """
    沿 z 轴查找在所有切片上均为 0 的背景区域，并在 xy 平面裁剪掉这部分背景。

    返回的体数据为原 volume 的子区域，以减少无效像素。
    """

    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array with shape (z, y, x)")

    vol = np.asarray(volume)
    mask_xy = np.any(vol != 0, axis=0)  # shape: (y, x)
    if not mask_xy.any():
        return vol.copy()

    y_mask = mask_xy.any(axis=1)
    x_mask = mask_xy.any(axis=0)
    y_indices = np.where(y_mask)[0]
    x_indices = np.where(x_mask)[0]

    y_start, y_end = y_indices[0], y_indices[-1] + 1
    x_start, x_end = x_indices[0], x_indices[-1] + 1

    return vol[:, y_start:y_end, x_start:x_end]


def show_3d(
    volume: np.ndarray,
    save_path: Path | str,
    percentile: float = 70.0,
    max_points: int = 200_000,
    elev: float = 30.0,
    azim: float = 35.0,
) -> Path:
    """
    将 3D 体数据可视化为 xyz 视角的 2D 图像，并保存到指定路径。

    Parameters
    ----------
    volume:
        输入的 3D 体数据，轴顺序应为 (z, y, x)。
    save_path:
        输出图像的路径（包含文件名）。父目录会自动创建。
    percentile:
        用于选取高亮体素的强度分位点，过滤噪声。
    max_points:
        绘制散点的最大数量，避免渲染过慢。
    elev, azim:
        matplotlib 3D 视角设置，默认展示 xyz 方向。
    """

    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array with shape (z, y, x)")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vol = np.asarray(volume)
    threshold = np.percentile(vol, percentile)
    mask = vol >= threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        coords = np.argwhere(vol == vol.max())

    intensities = vol[tuple(coords.T)]
    if coords.shape[0] > max_points:
        idx = np.linspace(0, coords.shape[0] - 1, max_points, dtype=np.int32)
        coords = coords[idx]
        intensities = intensities[idx]

    norm = Normalize(vmin=vol.min(), vmax=vol.max() if vol.max() > vol.min() else vol.min() + 1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords[:, 2],  # x
        coords[:, 1],  # y
        coords[:, 0],  # z
        c=norm(intensities),
        cmap="viridis",
        s=1,
        linewidths=0,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D Volume View (xyz)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path


def save_stack(
    volume: np.ndarray,
    save_path: Path | str,
    fmt: str | None = "tif",
    compression: str | None = "zlib",
) -> Path:
    """
    将 3D 体数据整体写出，可选择 tif 或 zarr 格式，默认 tif。

    Parameters
    ----------
    volume:
        需要保存的 3D numpy 数组。
    save_path:
        目标路径；对于 zarr 会创建同名目录。
    fmt:
        'tif' / 'tiff' / 'zarr'，若为 None 则根据文件后缀推断。
    compression:
        保存 tif 时使用的压缩方式，默认 zlib。
    """

    save_path = Path(save_path)
    if fmt is None:
        suffix = save_path.suffix.lower().lstrip(".")
        fmt = suffix or "tif"
    fmt = fmt.lower()

    if fmt in {"tif", "tiff"}:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(save_path, volume, compression=compression)
    elif fmt == "zarr":
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - informative guard
            raise RuntimeError(
                "zarr format requested but the zarr package is not installed."
            ) from exc

        if save_path.suffix == ".zarr":
            # zarr.save expects the directory not to exist yet
            if save_path.exists():
                # remove existing directory content
                import shutil

                shutil.rmtree(save_path)
        zarr.save(str(save_path), volume)
    else:
        raise ValueError("fmt must be one of {'tif', 'tiff', 'zarr'}")

    return save_path


def _normalize_size(
    size: int | Sequence[int],
    name: str,
) -> Tuple[int, int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"{name} must be positive.")
        return (size, size, size)
    if len(size) != 3:
        raise ValueError(f"{name} must contain exactly 3 elements.")
    z, y, x = (int(s) for s in size)
    if min(z, y, x) <= 0:
        raise ValueError(f"All elements of {name} must be positive.")
    return (z, y, x)


def extract_subvolume(
    volume: np.ndarray,
    patch_size: int | Sequence[int],
    start: Sequence[int] | None = None,
) -> np.ndarray:
    """
    根据指定起始坐标与尺寸，从 3D 体数据中截取一个 sub-volume。

    Parameters
    ----------
    volume:
        输入的三维数组，shape (z, y, x)。
    patch_size:
        sub-volume 的尺寸，单个整数表示立方块，或 (dz, dy, dx)。
    start:
        sub-volume 的起始坐标 (z, y, x)，默认 (0, 0, 0)。

    Returns
    -------
    np.ndarray
        截取得到的 sub-volume。
    """

    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array with shape (z, y, x)")

    vol = np.asarray(volume)
    pz, py, px = _normalize_size(patch_size, "patch_size")
    if start is None:
        z0 = y0 = x0 = 0
    else:
        if len(start) != 3:
            raise ValueError("start must contain exactly 3 elements (z, y, x).")
        z0, y0, x0 = (int(v) for v in start)
    if min(z0, y0, x0) < 0:
        raise ValueError("start indices must be non-negative.")

    z1, y1, x1 = z0 + pz, y0 + py, x0 + px
    if z1 > vol.shape[0] or y1 > vol.shape[1] or x1 > vol.shape[2]:
        raise ValueError(
            "Requested sub-volume exceeds original volume bounds: "
            f"start={(z0, y0, x0)}, size={(pz, py, px)}, "
            f"volume_shape={vol.shape}"
        )

    return vol[z0:z1, y0:y1, x0:x1]


def save_slices(
    volume: np.ndarray,
    output_dir: Path,
    reference_dtype: np.dtype,
    prefix: str = "slice",
) -> None:
    """沿 z 轴从上到下依次写出切片。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if np.issubdtype(reference_dtype, np.integer):
        dtype_info = np.iinfo(reference_dtype)
        clip_min, clip_max = dtype_info.min, dtype_info.max
    else:
        dtype_info = np.finfo(reference_dtype)
        clip_min, clip_max = dtype_info.min, dtype_info.max

    for idx, slice_arr in enumerate(volume):
        clipped = np.clip(slice_arr, clip_min, clip_max)
        if np.issubdtype(reference_dtype, np.integer):
            out_slice = np.rint(clipped).astype(reference_dtype)
        else:
            out_slice = clipped.astype(reference_dtype)
        out_path = output_dir / f"{prefix}_{idx:04d}.tif"
        tifffile.imwrite(out_path, out_slice, compression="zlib")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a 3D volume, rotate it around the yz plane, and export slices."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Volumes/T7/20251010_halfCell/preprocessedData_Part/6_flip_"),
        help="目录中应包含按顺序排列的 tiff 切片。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/T7/20251010_halfCell/preprocessedData_Part/7_RS"),
        help="旋转后切片的保存目录。",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=-36.0,
        help="绕 x 轴旋转的角度（度）。默认 -36°。",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rotated_slice",
        help="输出切片文件名前缀。",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    volume = read_volume(args.input_dir)
    save_stack(volume, args.output_dir / "volume_before1.tif")
    volume = delete_background(volume)
    save_stack(volume, args.output_dir / "volume_before2.tif")
    subvolume = extract_subvolume(volume, (3, 1000, 1000), (0, 1200, 600))
    save_stack(subvolume, args.output_dir / "volume_before3.tif")
    show_3d(subvolume, args.output_dir / "volume_before4.png")
    rotated = rotate_volume(subvolume, args.angle)
    show_3d(rotated, args.output_dir / "volume_after.png")
    save_slices(rotated, args.output_dir, volume.dtype, prefix=args.prefix)
    print(
        f"Rotated volume with angle {args.angle}° and exported "
        f"{rotated.shape[0]} slices to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
