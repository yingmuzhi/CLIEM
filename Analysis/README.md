# Analysis

This folder contains the analysis pipeline, plotting scripts, and the bundled toy example data used to generate the figures. The pipeline follows this flow:

1. **Convert predictions to instance labels** (`utils/convert.py`).
2. **Compute instance metrics** (`utils/metric_compute.py`).
3. **Plot per-class confusion matrices** (`utils/metric_plot.py`).
4. **Aggregate plots** across slices (`plot_whole_confusionFigure.py`, `plot_whole_boxFigure.py`).
5. **Plot summary statistics** from Excel reports (`plot_volume_surface_instance.py`, `plot_contactSites2.py`).

## Scripts

### `analysis_pipeline.py`
Runs the full pipeline for a single slice:
- Converts prediction masks to instance labels.
- Computes per-class metrics JSON.
- Plots per-class confusion matrices.

**Inputs**
- `--slice`: slice ID (used to build default paths).
- `--base_dir`: base data folder (defaults to `Analysis/src/toy_example`).
- Optional overrides: `--pred_dir`, `--pred_instances_dir`, `--gt_dir`, `--metrics_json`, `--plot_out_dir`.

**Outputs**
- Instance-label TIFFs in `prediction_instances/<slice>`.
- Metrics JSON `metrics_<slice>_iou<iou>.json`.
- Per-class confusion plots in `figures/confusion_plots_<slice>_iou<iou>`.

### `analysis_pipeline.sh`
Runs `analysis_pipeline.py` for multiple slices in sequence.

**Inputs**
- Uses `SLICES=("244" "288" "320" "348")` by default.
- Optional `PYTHON_BIN` environment variable to select Python.

**Outputs**
- Same as `analysis_pipeline.py`, repeated for each slice.

### `pipeline.ipynb`
Notebook that runs the full analysis workflow in order and displays the generated outputs.

**Runs**
- `analysis_pipeline.py` (all slices)
- `plot_whole_boxFigure.py`
- `plot_whole_confusionFigure.py`
- `plot_volume_surface_instance.py`
- `plot_contactSites2.py`

**Outputs**
- Displays one metrics JSON preview.
- Renders box plot, merged heatmap, volume/instance/surface plot, and contact-site plot inline.

### `plot_whole_confusionFigure.py`
Builds a merged heatmap figure (organelle × slice × metric blocks) from multiple metrics JSON files.

**Inputs**
- `--metrics`: list of `metrics_*.json` files.

**Outputs**
- `confusion_summary_iou*.png` and optional PDF.

### `plot_whole_boxFigure.py`
Creates boxplots of Precision/Recall/F1/Boundary F1 across slices for each organelle.

**Inputs**
- `--metrics`: list of `metrics_*.json` files.

**Outputs**
- `box_metrics_iou*.png` and optional PDF.

### `plot_volume_surface_instance.py`
Plots stacked bars for **Volume**, **Instance number**, and **Surface area** using organelle statistics from an Excel report.

**Inputs**
- `--excel`: Excel file (e.g. `analysis4_OnePageMultiComponent_Average.xls`).
- `--cell-volume-um3`: whole-cell volume (used for Volume labels).

**Outputs**
- `volume_instance_surface_stacked.png`.

### `plot_contactSites2.py`
Plots stacked bars for **Instance number** and **Surface area** for ER contact sites.

**Inputs**
- `--input`: Excel file (e.g. `analysis6_OnePageMultiComponent_Average.xls`).

**Outputs**
- `contact_sites_combined.png`.

## Utilities

### `utils/convert.py`
Converts prediction masks to instance-label TIFFs via connected components.

**Inputs**
- `--input_dir`: prediction masks directory.
- `--output_dir`: output instance labels.

**Outputs**
- Instance-label TIFFs with labels 0..N.

### `utils/metric_compute.py`
Computes instance-level metrics and boundary F1 from prediction instances vs ground truth.

**Inputs**
- `--pred_dir`, `--gt_dir`.
- `--iou`, `--boundary_tol`.

**Outputs**
- Metrics JSON (`metrics_<slice>_iou<iou>.json`).

### `utils/metric_plot.py`
Plots per-class confusion-style heatmaps from a metrics JSON file.

**Inputs**
- `--metrics_json`.
- `--out_dir`.

**Outputs**
- `confusion_<class>.png` per organelle.
- `confusion_plots_index.json`.

### `utils/inspect_xls.py`
Quick utility to dump an Excel sheet to stdout for inspection.

**Inputs**
- Hardcoded Excel path inside the script.

**Outputs**
- Printed table in the terminal.

## Data (toy example)

### `src/toy_example/`
Bundled sample data used by the scripts by default. Key subfolders:

- `prediction/` and `prediction_instances/`: prediction masks and instance labels.
- `ground_truth/`: GT instance labels.
- `metrics_*.json`: metrics outputs for each slice.
- `figures/`: generated plots (boxplots, confusion plots, stacked bars).
- `volume_surf_ins_src/` and `contactSites_src/`: Excel summary reports.
  - These reports are generated from **Imaris 11.0** after **surface fitting** to compute distances and contact sites.

> The toy example is meant for demonstration and validation. Replace `--base_dir`, `--metrics`, or Excel paths to run on your own dataset.
