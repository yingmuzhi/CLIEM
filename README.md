# Preprocess (SEM) — CLIEM_code

This repository contains a set of scripts under `Preprocess/` to preprocess SEM image slices (typically exported as per-slice TIFFs, grouped by parts) and generate a corrected, reconstructed stack ready for downstream analysis.

Most scripts ship with **dataset-specific default paths** (e.g. `/Volumes/T7/...`). In practice you should **always pass your own arguments** when running them.

## Quick start (recommended)

### 1) Create a Python environment

Use Python 3.9+ (3.11 recommended). Install all dependencies with one command:

```bash
pip install -e .
```

**Docker** and **Anaconda** are also recommended.

### 2) Prepare an input folder (`base_dir`)

The wrapper pipeline (`Preprocess/pipeline.py`) is easiest to use if you structure your data like this:

```text
<base_dir>/
  1_align/
    part1/
      *.tif
      Results.csv
    part2/
      *.tif
      Results.csv
    ...
```

Notes:
- `part*` folders must contain the per-slice TIFF files.
- Each part should contain a `Results.csv` with per-slice drift values (`Slice`, `dX`, `dY`).

### 3) Run the end-to-end pipeline

```bash
python Preprocess/pipeline.py --base-dir "<base_dir>"
```

This will run:

1. `pad_image.py`
2. `drift_correction.py` (wraps `drift_part.py`)
3. `merge_stack.py`
4. `reconstruct_sem.py`
5. `flip_sem.py`
6. Post-step: `delete_redundant_BG.py` (cropping redundant zero background)

### 4) Outputs

When you pass `--base-dir`, `pipeline.py` auto-creates stage paths:

```text
<base_dir>/
  1_align/                 (input)
  2_pad/                   (padded slices + padding_info.json)
  3_drift/                 (drift-corrected slices)
  4_merge/                 (merged/streamed slices; see note below)
  5_reconstruct/           (reconstructed slices + metadata.txt)
  6_flip/                  (flipped slices + flip_metadata.txt)
  7_deleteBG/              (cropped slices; tight bounding box across stack)
```

**Important (merge output):** `Preprocess/merge_stack.py` writes **individual 2D slices** named like `slice_000000.tif` into an output *directory*. The pipeline passes a path ending in `drift_merged.tif` by default, but `merge_stack.py` treats `--output-file` as a legacy alias for an output directory. The next stage (`reconstruct_sem.py`) can process either a **directory of 2D slices** or a **single 3D TIFF**, so the default pipeline still works.

## What each stage does (logic flow)

### 1) Padding (`pad_image.py`)

- **Input:** `<base_dir>/1_align/part*/` (2D TIFF slices; may have varying canvas sizes)
- **Output:** `<base_dir>/2_pad/part*/` (padded slices, same target canvas)
- **Also writes:** `padding_info.json` (per-file padding) and **copies each part’s `Results.csv`** into the matching output folder.
- **Modes:** `auto` (pad to max), `manual`, `target_size`.

Typical manual/target-size usage:

```bash
python Preprocess/pad_image.py \
  --input "<base_dir>/1_align" \
  --output_dir "<base_dir>/2_pad" \
  --mode manual \
  --pad_left 150 --pad_right 150 --pad_top 500 --pad_bottom 1500
```

### 2) Drift correction (`drift_correction.py` → `drift_part.py`)

High-level wrapper:
- **Input:** a directory containing `part*` subfolders (default: padded output).
- **For each part:** uses `<part>/Results.csv` to translate each slice (black fill).
- **Multiple parts:** computes cross-part “gap” offsets from each part’s `Results.csv`, writes `gap_results.csv`, and **adds cumulative offsets into downstream parts’ `Results.csv` (in-place)** before correcting.

Run via wrapper:

```bash
python Preprocess/drift_correction.py \
  --input-dir "<base_dir>/2_pad" \
  --output-dir "<base_dir>/3_drift" \
  --threshold 50 \
  --drift-gap True
```

Direct per-part run (if you want to bypass the wrapper):

```bash
python Preprocess/drift_part.py \
  --input-dir "<base_dir>/2_pad/part1" \
  --output-dir "<base_dir>/3_drift/part1" \
  --transformations "<base_dir>/2_pad/part1/Results.csv" \
  --threshold 50
```

### 3) Merge / re-index (`merge_stack.py`)

This step “linearizes” all drift-corrected TIFF files (recursively) and writes them out as sequential slices:

```bash
python Preprocess/merge_stack.py \
  --input-dir "<base_dir>/3_drift" \
  --output-dir "<base_dir>/4_merge"
```

You will get:
- `slice_000000.tif`, `slice_000001.tif`, ...
- `merge_metadata.txt`

### 4) Reconstruction (`reconstruct_sem.py`)

Per-slice reconstruction:
- stretches each slice vertically by \(1/\sin(A)\)
- adds slice-dependent padding based on angle \(A\), \(\tan(A)\), and `--votex-size` (voxel size / pixel size ratio)
- processes data **lazily slice-by-slice** when input is a directory (memory-friendly)

Directory input (recommended with the default pipeline):

```bash
python Preprocess/reconstruct_sem.py \
  --input-file "<base_dir>/4_merge" \
  --output-path "<base_dir>/5_reconstruct" \
  --angle 54 \
  --interpolation cubic \
  --votex-size "(2.458, 2.458, 8)"
```

### 5) Flip (`flip_sem.py`)

Flips each output slice in both axes:

```bash
python Preprocess/flip_sem.py \
  --input-dir "<base_dir>/5_reconstruct" \
  --output-dir "<base_dir>/6_flip"
```

### 6) Crop redundant background (`delete_redundant_BG.py`)

Computes a tight bounding box across the entire stack (non-zero pixels) and crops all slices consistently:

```bash
python -c "from Preprocess.delete_redundant_BG import delete_redundant_background; delete_redundant_background('<base_dir>/6_flip','<base_dir>/7_deleteBG')"
```

(`pipeline.py` runs this automatically after `flip_sem.py`.)

## 3D “stack” variants (optional)

Some `*_20251125_3Dversion.py` scripts are alternative implementations that more explicitly handle 3D stacks or earlier assumptions:
- `pipeline_20251125_3Dversion.py`
- `drift_part_20251125_3Dversion.py`
- `merge_stack_20251125_3Dversion.py`
- `reconstruct_sem_20251125_3Dversion.py`

If you need a single multi-page `drift_merged.tif`, use:

```bash
python Preprocess/merge_stack_20251125_3Dversion.py \
  --input-dir "<base_dir>/3_drift" \
  --output-file "<base_dir>/4_merge/drift_merged.tif"
```

## Utilities (run as needed)

- `crop_manual.py`: crop each 2D/3D TIFF by fixed pixel margins (multi-process).
- `extract_slice.py`: copy every *n*-th slice to a new folder (multi-process).
- `manual_matrix.py`: generate a `Results.csv` from user-provided `dx/dy` expressions.
- `calculate_imageMinMax.py`: compute min/max width/height across a list of TIFFs.
- `calculate_resultCSVMinMax.py`: compute min/max `dX/dY` across multiple `Results.csv`.
- `seperate_into_parts.py`: split a folder of images into `part1..part4` by index ranges and copy `Results.csv`.
- `seperate_specificImageGap.py`: export boundary images between consecutive parts (“gap images”).
- `drift_gap.py`: apply one or more `Results.csv` files as a drift correction to a 3D stack or a directory.
- `rotate_sample.py`: rotate a 3D volume around the yz plane and export slices (also has visualization helpers).
- `add_offset.py`: add a constant offset to `dY` in a `Results.csv` (one-off fix).
- `create_test_data.py`: generate a small synthetic 3D TIFF for testing `reconstruct_sem.py`.

## Batch runner (dataset-specific)

`Preprocess/run_pipeline_all.sh` loops through subfolders under a hard-coded `TARGET_ROOT` and calls `pipeline.py` via a conda environment. You will likely need to edit `TARGET_ROOT`, `PIPELINE_SCRIPT`, and `CONDA_ENV` before using it.

## Legacy scripts

`Preprocess/legacy/` contains older prototypes and “copy” variants. They are kept for reference and reproducibility, but the recommended entrypoint is `Preprocess/pipeline.py` (or the `*_3Dversion.py` scripts if you explicitly need them).

