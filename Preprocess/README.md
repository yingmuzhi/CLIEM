## Preprocess pipeline (`pipeline.py`)

This directory contains the end‑to‑end SEM preprocessing pipeline and related utilities.
The main entrypoint is `pipeline.py`, which orchestrates:

1. `pad_image.py`
2. `drift_correction.py` → `drift_part.py`
3. `merge_stack.py`
4. `reconstruct_sem.py`
5. `flip_sem.py`
6. `delete_redundant_BG.py` (called as a Python function, not a subprocess)

This document focuses on how to run `pipeline.py` and how to set the key arguments:

- **data root**: `--base-dir`
- **padding pixels** (canvas size and margins)
- **reconstruction angle**: `--recon-angle`
- **voxel / pixel size**: `--recon-votex-size`

---

## 1. Required directory layout (`--base-dir`)

The pipeline expects a *base directory* with this structure:

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
  2_pad/          # created by the pipeline
  3_drift/
  4_merge/
  5_reconstruct/
  6_flip/
  7_deleteBG/
```

Each `part*` folder under `1_align` must contain:

- per‑slice TIFF images
- a `Results.csv` with at least the columns `Slice`, `dX`, `dY`

### `--base-dir`

- **Purpose**: single root for the whole dataset.
- **Type**: absolute or relative filesystem path.

When `--base-dir` is provided, `pipeline.py` automatically derives all stage paths:

- `pad_input      = <base_dir>/1_align`
- `pad_output     = <base_dir>/2_pad`
- `drift_input    = pad_output`
- `drift_output   = <base_dir>/3_drift`
- `merge_input    = drift_output`
- `merge_output   = <base_dir>/4_merge/drift_merged.tif`
- `recon_input    = merge_output`
- `recon_output   = <base_dir>/5_reconstruct`
- `flip_input     = recon_output`
- `flip_output    = <base_dir>/6_flip`
- `delete_bg_output = <base_dir>/7_deleteBG`

**Recommended minimal command:**

```bash
python Preprocess/pipeline.py \
  --base-dir "/absolute/path/to/your_dataset"
```

You normally do **not** need to override individual per‑stage paths if the directory
layout follows the convention above.

---

## 2. Padding configuration (canvas size & margins)

Padding is handled by `pad_image.py`. In `pipeline.py` all related options are
prefixed with `--pad-`:

- `--pad-mode {auto,manual,target_size}` (default: `manual`)
- `--pad-input` / `--pad-output`
- `--pad-left`, `--pad-right`, `--pad-top`, `--pad-bottom`
- `--pad-target-width`, `--pad-target-height`
- `--pad-lzw`, `--no-pad-lzw`

When `--base-dir` is set:

- `pad_input  = <base_dir>/1_align`
- `pad_output = <base_dir>/2_pad`

### 2.1 `--pad-mode`

- **`auto`**  
  Compute the maximum width/height across all input images and pad smaller images
  up to that maximum. Margins are implicit.

- **`manual`** (recommended for most use cases)  
  You directly specify how many pixels to add on each side; the target canvas
  is computed as:

  \[
  \text{target\_width}  = \text{max\_width}  + \text{pad\_left} + \text{pad\_right}
  \]
  \[
  \text{target\_height} = \text{max\_height} + \text{pad\_top}  + \text{pad\_bottom}
  \]

- **`target_size`**  
  You specify the final canvas size and (optionally) fixed margins; the script
  adjusts right/bottom padding so that all images fit exactly into
  `--pad-target-width × --pad-target-height`.

### 2.2 How to choose padding values

Typical manual configuration:

```bash
python Preprocess/pipeline.py \
  --base-dir "/absolute/path/to/your_dataset" \
  --pad-mode manual \
  --pad-left 150  --pad-right 150 \
  --pad-top 500   --pad-bottom 1500
```

Guidelines:

- Use **visual inspection** (e.g. in Fiji/ImageJ) to decide how much margin
  you need on each side to accommodate drift and reconstruction padding later.
- If you already know a desired final canvas size (e.g. from a previous run),
  use `target_size` with explicit `--pad-target-width` and `--pad-target-height`.

---

## 3. Reconstruction angle (`--recon-angle`)

Reconstruction is performed by `reconstruct_sem.py`. The key geometric parameter
is the angle \(A\) (in degrees), used to:

- vertically stretch each slice by factor \(1 / \sin(A)\)
- compute per‑slice top/bottom padding based on \(\tan(A)\) and voxel size

In `pipeline.py` this is exposed as:

- `--recon-angle` (default: `54.0`)

Example:

```bash
python Preprocess/pipeline.py \
  --base-dir "/absolute/path/to/your_dataset" \
  --recon-angle 54
```

**How to choose the angle:**

- Use the **physical tilt angle** of the sample or beam in your SEM acquisition.
- Typical values are in the range **40–60°**. A larger angle means larger
  vertical stretch and different padding geometry.

---

## 4. Voxel / pixel size (`--recon-votex-size`)

To make the reconstruction geometry consistent with real physical spacing,
`reconstruct_sem.py` takes a voxel size parameter:

- Pipeline flag: `--recon-votex-size`
- Passed through to `reconstruct_sem.py` as a string representing a tuple:

  ```text
  "(vx, vy, vz)"
  ```

  where:

  - `vx`, `vy` = in‑plane pixel size (e.g. nm/px)
  - `vz`       = slice spacing / thickness (e.g. nm)

Default in `pipeline.py`:

- `--recon-votex-size "(2.458, 2.458, 8)"`

Internally, the script computes:

\[
z\_ratio = \frac{vz}{vx}
\]

and uses this in the padding formulas so that the apparent geometry matches
physical distances more closely.

### 4.1 How to fill in `--recon-votex-size`

1. Read the pixel size from your **microscope metadata** or acquisition settings:
   - e.g. 2.458 nm/px in X/Y
   - 8 nm between slices in Z
2. Supply them as a tuple string:

   ```bash
   --recon-votex-size "(2.458, 2.458, 8)"
   ```

Constraints:

- `vx` and `vy` must be equal (the code checks this).
- All three values must be positive numbers.

If you are unsure, you can start with the default and adjust later once you
have confirmed the correct voxel size.

---

## 5. Putting it all together (example command)

A realistic example that configures only the most important parameters:

```bash
python Preprocess/pipeline.py \
  --base-dir "/absolute/path/to/your_dataset" \
  --pad-mode manual \
  --pad-left 150  --pad-right 150 \
  --pad-top 500   --pad-bottom 1500 \
  --recon-angle 54 \
  --recon-votex-size "(2.458, 2.458, 8)"
```

This will:

1. Read raw aligned slices from `<base_dir>/1_align/part*/`.
2. Write padded slices and `padding_info.json` into `<base_dir>/2_pad/`.
3. Apply drift correction into `<base_dir>/3_drift/`.
4. Linearize slices into `<base_dir>/4_merge/drift_merged.tif` (or directory).
5. Reconstruct slices into `<base_dir>/5_reconstruct/`.
6. Flip slices into `<base_dir>/6_flip/`.
7. Crop redundant background into `<base_dir>/7_deleteBG/`.

Once this command completes without errors, the directory `<base_dir>/7_deleteBG`
contains the final, tightly cropped stack ready for downstream analysis.

