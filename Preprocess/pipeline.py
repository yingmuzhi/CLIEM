# !/usr/bin/env python3
"""
End-to-end pipeline runner for the SEM preprocessing workflow.

This script orchestrates the following stages located in the same directory:
1. pad_image.py
2. drift_correction.py
3. merge_stack.py
4. reconstruct_sem.py
5. flip_sem.py

All stage-specific parameters can be provided once through this wrapper.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from .delete_redundant_BG import delete_redundant_background


def bool_to_str(value: bool) -> str:
    """Convert bool to the string literals expected by scripts using type=bool."""
    return "True" if value else "False"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pad → drift → merge → reconstruct → flip pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    script_dir = Path(__file__).resolve().parent

    general = parser.add_argument_group("General")
    general.add_argument(
        "--base-dir",
        default="/Volumes/T7/20251204_halfCell/CLEM_code/Preprocess/src/toy_example",
        # default="/Volumes/T7/20251204_halfCEll/src/code_analysis2",
        help="Base directory for all pipeline stages. If provided, automatically generates:\n"
             "  - pad_input: {base_dir}/1_align\n"
             "  - pad_output: {base_dir}/2_pad\n"
             "  - drift_output: {base_dir}/3_drift\n"
             "  - merge_output: {base_dir}/4_merge/drift_merged.tif\n"
             "  - recon_output: {base_dir}/5_reconstruct\n"
             "  - flip_output: {base_dir}/6_flip\n"
             "  - delete_bg_output: {base_dir}/7_deleteBG\n"
             "Individual stage paths can still be overridden if needed.",
    )
    general.add_argument(
        "--scripts-root",
        default=str(script_dir),
        help="Directory that contains the stage scripts",
    )
    general.add_argument(
        "--delete-bg-output",
        default="/Volumes/DATA5/src/7_deleteBG",
        help="Output directory for slices after delete_background runs on the flipped stack",
    )

    pad = parser.add_argument_group("pad_image.py")
    pad.add_argument(
        "--pad-input",
        # default="/Volumes/T7/20251010_halfCell/preprocessedData_Part/1_align",
        default="/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/2_part",
        help="Input root directory that contains part subfolders",
    )
    pad.add_argument(
        "--pad-output",
        default="/Volumes/DATA5/src/2_pad",
        help="Output directory for padded images",
    )
    pad.add_argument(
        "--pad-mode",
        choices=["auto", "manual", "target_size"],
        default="manual",
        help="Padding mode",
    )
    pad.add_argument("--pad-target-width", type=int, default=0)
    pad.add_argument("--pad-target-height", type=int, default=0)
    pad.add_argument("--pad-left", type=int, default=150)
    pad.add_argument("--pad-right", type=int, default=150)
    pad.add_argument("--pad-top", type=int, default=500)
    pad.add_argument("--pad-bottom", type=int, default=1500)
    pad.add_argument(
        "--pad-lzw",
        action="store_true",
        default=True,
        help="Enable LZW compression for pad_image (use --no-pad-lzw to disable)",
    )
    pad.add_argument(
        "--no-pad-lzw",
        action="store_true",
        help="Disable LZW compression for pad_image",
    )

    drift = parser.add_argument_group("drift_correction.py")
    drift.add_argument(
        "--drift-input",
        help="Input directory for drift correction (defaults to pad output)",
    )
    drift.add_argument(
        "--drift-output",
        default="/Volumes/DATA5/src/3_drift",
        help="Output directory for drift correction results",
    )
    drift.add_argument("--drift-pattern", default="*.tif")
    drift.add_argument(
        "--drift-bit-depth",
        choices=["8", "16"],
        default="8",
    )
    drift.add_argument(
        "--drift-lzw",
        action="store_true",
        default=True,
        help="Enable LZW compression (use --no-drift-lzw to disable)",
    )
    drift.add_argument(
        "--no-drift-lzw",
        action="store_true",
        help="Disable LZW compression for drift stage",
    )
    drift.add_argument("--drift-verbose", action="store_true", default=True)
    drift.add_argument(
        "--drift-threshold",
        type=float,
        default=50.0,
    )
    drift.add_argument("--drift-gap", action="store_true", default=True)
    drift.add_argument(
        "--no-drift-gap", action="store_true", help="Disable drift gap correction"
    )

    merge = parser.add_argument_group("merge_stack.py")
    merge.add_argument(
        "--merge-input",
        help="Input directory for merge stack (defaults to drift output)",
    )
    merge.add_argument(
        "--merge-output",
        default="/Volumes/DATA5/src/4_merge",
        # default="/Volumes/DATA5/src/4_merge/drift_merged.tif",
        help="Output TIF file path for merged stack",
    )
    merge.add_argument(
        "--merge-bit-depth",
        choices=["8", "16"],
        default="8",
    )
    merge.add_argument(
        "--merge-lzw",
        action="store_true",
        default=False,
        help="Enable LZW compression for merge_stack",
    )
    merge.add_argument("--merge-verbose", action="store_true", default=True)

    recon = parser.add_argument_group("reconstruct_sem.py")
    recon.add_argument(
        "--recon-input",
        help="Input merged TIF file (defaults to merge output)",
    )
    recon.add_argument(
        "--recon-output",
        default="/Volumes/DATA5/src/5_reconstruct",
        help="Output directory for reconstructed slices",
    )
    recon.add_argument("--recon-angle", type=float, default=54.0)
    recon.add_argument(
        "--recon-interp",
        choices=["nearest", "linear", "cubic"],
        default="cubic",
    )
    recon.add_argument(
        "--recon-bit-depth",
        choices=["8", "16"],
        default="8",
    )
    recon.add_argument(
        "--recon-lzw",
        action="store_true",
        default=True,
        help="Enable LZW compression (use --no-recon-lzw to disable)",
    )
    recon.add_argument(
        "--no-recon-lzw",
        action="store_true",
        help="Disable LZW compression for reconstruction",
    )
    recon.add_argument("--recon-verbose", action="store_true")
    recon.add_argument(
        "--recon-votex-size",
        default="(2.458, 2.458, 8)",
        # default="(2.458, 2.458, 8)",
        help="Tuple string '(x, y, z)' passed to reconstruct_sem.py",
    )

    flip = parser.add_argument_group("flip_sem.py")
    flip.add_argument(
        "--flip-input",
        help="Input directory for flip stage (defaults to reconstruction output)",
    )
    flip.add_argument(
        "--flip-output",
        default="/Volumes/DATA5/src/6_flip",
        help="Output directory for flipped slices",
    )
    flip.add_argument("--flip-pattern", default="*.tif")
    flip.add_argument(
        "--flip-bit-depth",
        choices=["8", "16"],
        default="8",
    )
    flip.add_argument(
        "--flip-lzw",
        action="store_true",
        default=True,
        help="Enable LZW compression (use --no-flip-lzw to disable)",
    )
    flip.add_argument(
        "--no-flip-lzw",
        action="store_true",
        help="Disable LZW compression for flip stage",
    )
    flip.add_argument("--flip-verbose", action="store_true")

    return parser


def resolve_paths(args):
    """Resolve dependent defaults across stages.
    
    If --base-dir is provided, automatically generates all stage paths.
    Otherwise, uses individual path arguments or their defaults.
    """
    base_dir = getattr(args, 'base_dir', None)
    
    if base_dir:
        # Normalize base_dir path
        base_path = Path(base_dir).resolve()
        
        print(f"Using base directory: {base_path}")
        print("Auto-generating stage paths:")
        
        # Auto-generate all paths from base_dir
        args.pad_input = str(base_path / "1_align")
        args.pad_output = str(base_path / "2_pad")
        args.drift_output = str(base_path / "3_drift")
        args.merge_output = str(base_path / "4_merge" / "drift_merged.tif")
        args.recon_output = str(base_path / "5_reconstruct")
        args.flip_output = str(base_path / "6_flip")
        args.delete_bg_output = str(base_path / "7_deleteBG")
        
        print(f"  pad_input:     {args.pad_input}")
        print(f"  pad_output:    {args.pad_output}")
        print(f"  drift_output:  {args.drift_output}")
        print(f"  merge_output:  {args.merge_output}")
        print(f"  recon_output:  {args.recon_output}")
        print(f"  flip_output:   {args.flip_output}")
        print(f"  delete_bg_output: {args.delete_bg_output}")
        
        # Set dependent inputs (these are always derived from previous outputs)
        args.drift_input = args.pad_output
        args.merge_input = args.drift_output
        args.recon_input = args.merge_output
        args.flip_input = args.recon_output
    else:
        # Original behavior: resolve dependent defaults
        if not args.drift_input:
            args.drift_input = args.pad_output
        if not args.merge_input:
            args.merge_input = args.drift_output
        if not args.recon_input:
            args.recon_input = args.merge_output
        if not args.flip_input:
            args.flip_input = args.recon_output
        if not getattr(args, "delete_bg_output", None):
            args.delete_bg_output = str(Path(args.flip_output).parent / "7_deleteBG")
    
    return args


def stage_command(args, scripts_root: Path):
    pad_cmd = [
        sys.executable,
        str(scripts_root / "pad_image.py"),
        "--input",
        args.pad_input,
        "--output_dir",
        args.pad_output,
        "--mode",
        args.pad_mode,
        "--target_width",
        str(args.pad_target_width),
        "--target_height",
        str(args.pad_target_height),
        "--pad_left",
        str(args.pad_left),
        "--pad_right",
        str(args.pad_right),
        "--pad_top",
        str(args.pad_top),
        "--pad_bottom",
        str(args.pad_bottom),
        "--lzw-compression",
        bool_to_str(args.pad_lzw and not args.no_pad_lzw),
    ]

    drift_cmd = [
        sys.executable,
        str(scripts_root / "drift_correction.py"),
        "--input-dir",
        args.drift_input,
        "--output-dir",
        args.drift_output,
        "--pattern",
        args.drift_pattern,
        "--bit-depth",
        args.drift_bit_depth,
        "--lzw-compression",
        bool_to_str(args.drift_lzw and not args.no_drift_lzw),
        "--verbose",
        bool_to_str(args.drift_verbose),
        "--threshold",
        str(args.drift_threshold),
        "--drift-gap",
        bool_to_str(args.drift_gap and not args.no_drift_gap),
    ]

    merge_cmd = [
        sys.executable,
        str(scripts_root / "merge_stack.py"),
        "--input-dir",
        args.merge_input,
        "--output-file",
        args.merge_output,
        "--bit-depth",
        args.merge_bit_depth,
        "--lzw-compression",
        bool_to_str(args.merge_lzw),
        "--verbose",
        bool_to_str(args.merge_verbose),
    ]

    recon_cmd = [
        sys.executable,
        str(scripts_root / "reconstruct_sem.py"),
        "--input-file",
        args.recon_input,
        "--output-path",
        args.recon_output,
        "--angle",
        str(args.recon_angle),
        "--interpolation",
        args.recon_interp,
        "--bit-depth",
        args.recon_bit_depth,
        "--votex-size",
        args.recon_votex_size,
    ]
    if args.recon_verbose:
        recon_cmd.append("--verbose")
    if args.recon_lzw and not args.no_recon_lzw:
        recon_cmd.append("--lzw-compression")
    else:
        recon_cmd.append("--no-lzw-compression")

    flip_cmd = [
        sys.executable,
        str(scripts_root / "flip_sem.py"),
        "--input-dir",
        args.flip_input,
        "--output-dir",
        args.flip_output,
        "--pattern",
        args.flip_pattern,
        "--bit-depth",
        args.flip_bit_depth,
    ]
    if args.flip_lzw and not args.no_flip_lzw:
        flip_cmd.append("--lzw-compression")
    else:
        flip_cmd.append("--no-lzw-compression")
    if args.flip_verbose:
        flip_cmd.append("--verbose")

    return [
        ("pad_image.py", pad_cmd),
        ("drift_correction.py", drift_cmd),
        ("merge_stack.py", merge_cmd),
        ("reconstruct_sem.py", recon_cmd),
        ("flip_sem.py", flip_cmd),
    ]


def run_pipeline(commands, post_stage_hooks=None):
    for label, cmd in commands:
        # === skipping commands ===
        # if label == "pad_image.py" or label == "drift_correction.py" or label == "merge_stack.py":
        #     print(f"\n=== Skipping {label} ===")
        #     continue
        # === skipping commands ===

        print(f"\n=== Running {label} ===")
        print("Command:", " ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(
                f"{label} failed with exit code {result.returncode}. Pipeline aborted."
            )

        if post_stage_hooks and label in post_stage_hooks:
            post_stage_hooks[label]()

    print("\nPipeline completed successfully.")


def main():
    parser = build_parser()
    args = resolve_paths(parser.parse_args())

    # Normalize booleans for pad/drift defaults
    args.pad_lzw = not args.no_pad_lzw if args.no_pad_lzw else args.pad_lzw
    args.drift_lzw = not args.no_drift_lzw if args.no_drift_lzw else args.drift_lzw
    args.drift_gap = not args.no_drift_gap if args.no_drift_gap else args.drift_gap

    scripts_root = Path(args.scripts_root).resolve()
    commands = stage_command(args, scripts_root)
    flip_uses_lzw = args.flip_lzw and not args.no_flip_lzw
    post_hooks = {
        "flip_sem.py": lambda: delete_redundant_background(
            args.flip_output,
            args.delete_bg_output,
            pattern=args.flip_pattern,
            use_lzw=flip_uses_lzw,
        )
    }
    run_pipeline(commands, post_stage_hooks=post_hooks)


if __name__ == "__main__":
    main()

