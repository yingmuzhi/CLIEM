#!/usr/bin/env python3
"""
Manual Matrix Generation Script
This script generates a Results.csv file with manually configured dx and dy values
based on user-defined functions for each slice.
"""

import argparse
import os
import csv
import glob
import numpy as np
from pathlib import Path
import natsort
from typing import Callable, Optional


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Results.csv file with manually configured dx and dy values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with input directory (num-slices auto-detected)
  python manual_matrix.py --input-dir /path/to/tif/folder --output Results.csv --dx-func "slice * 2" --dy-func "slice * 3"
  
  # With custom functions
  python manual_matrix.py --input-dir /path/to/tif/folder --output Results.csv --start-slice 1 --dx-func "-6 + slice * -0.5" --dy-func "3 + slice * 0.3"
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default="/Volumes/T7/20251010_halfCell/preprocessedData_Part/part4_2",
        help='Input directory containing TIF files (num-slices will be auto-detected from file count)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/part4_2/Results.csv',
        help='Output CSV file path (default: Results.csv)'
    )
    
    parser.add_argument(
        '--start-slice',
        type=int,
        default=1,
        help='Starting slice number (default: 2)'
    )
    
    parser.add_argument(
        '--dx-func',
        type=str,
        default='0', #'-6 + slice * -0.1',
        help='Function expression for dx calculation. Use "slice" as variable. Example: "slice * 2" or "-6 + slice * -0.1" (default: "-6 + slice * -0.1")'
    )
    
    parser.add_argument(
        '--dy-func',
        type=str,
        default='0 + slice * 2.5',
        help='Function expression for dy calculation. Use "slice" as variable. Example: "slice * 3" or "3 + slice * 0.2" (default: "3 + slice * 0.2")'
    )
    
    parser.add_argument(
        '--use-round',
        type=bool,
        default=True,
        help='Whether to round dx and dy values to integers. Set to True to enable rounding, False to disable (default: True). Note: This parameter cannot be manually input from command line.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=bool,
        default=True,
        help='Enable verbose output. Set to True to enable, False to disable (default: True). Note: This parameter cannot be manually input from command line.'
    )
    
    return parser.parse_args()


def count_tif_files(input_dir: str, pattern: str = '*.tif', verbose: bool = False) -> int:
    """
    Count TIF files in the specified directory.
    
    Args:
        input_dir (str): Path to directory containing TIF files
        pattern (str): File pattern to match (default: '*.tif')
        verbose (bool): Enable verbose output
    
    Returns:
        int: Number of TIF files found
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find all TIF files matching the pattern
    search_pattern = os.path.join(input_dir, pattern)
    tif_files = glob.glob(search_pattern)
    
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {input_dir} matching pattern '{pattern}'")
    
    # Sort files naturally using natsort
    sorted_files = natsort.natsorted(tif_files)
    
    if verbose:
        print(f"Found {len(sorted_files)} TIF files in {input_dir}")
        print("Files sorted naturally:")
        for i, file_path in enumerate(sorted_files[:5]):
            print(f"  {i+1}: {os.path.basename(file_path)}")
        if len(sorted_files) > 5:
            print(f"  ... and {len(sorted_files) - 5} more files")
    
    return len(sorted_files)


def evaluate_function(func_str: str, slice_num: int) -> float:
    """
    Evaluate a function expression with slice as variable.
    
    Args:
        func_str (str): Function expression string (e.g., "slice * 2")
        slice_num (int): Current slice number
    
    Returns:
        float: Evaluated result
    """
    try:
        # Create a safe evaluation context
        safe_dict = {
            'slice': float(slice_num),
            'np': np,
            'int': int,
            'float': float,
            'round': round,
            'abs': abs,
            'min': min,
            'max': max,
        }
        
        # Evaluate the expression
        result = eval(func_str, {"__builtins__": {}}, safe_dict)
        return float(result)
    
    except Exception as e:
        raise ValueError(f"Error evaluating function '{func_str}': {e}")


def generate_results_csv(output_path: str,
                         start_slice: int,
                         num_slices: int,
                         dx_func: str,
                         dy_func: str,
                         round_to_int: bool = True,
                         verbose: bool = False):
    """
    Generate Results.csv file with dx and dy values calculated from functions.
    
    Args:
        output_path (str): Output CSV file path
        start_slice (int): Starting slice number
        num_slices (int): Number of slices to generate
        dx_func (str): Function expression for dx calculation
        dy_func (str): Function expression for dy calculation
        round_to_int (bool): Whether to round values to integers
        verbose (bool): Enable verbose output
    """
    if verbose:
        print(f"Generating Results.csv file...")
        print(f"  Output path: {output_path}")
        print(f"  Start slice: {start_slice}")
        print(f"  Number of slices: {num_slices}")
        print(f"  dx function: {dx_func}")
        print(f"  dy function: {dy_func}")
        print(f"  Round to int: {round_to_int}")
        print()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    # Generate data
    rows = []
    rows.append(['', 'Slice', 'dX', 'dY'])  # Header row
    
    for i in range(num_slices):
        slice_num = start_slice + i
        
        # Calculate dx and dy using the provided functions
        dx = evaluate_function(dx_func, slice_num)
        dy = evaluate_function(dy_func, slice_num)
        
        # Round to integers if requested
        if round_to_int:
            dx = int(round(dx))
            dy = int(round(dy))
        
        # Add row: index (1-based), slice number, dx, dy
        rows.append([i + 1, slice_num, dx, dy])
        
        if verbose and (i < 5 or i == num_slices - 1):
            print(f"  Slice {slice_num}: dx={dx}, dy={dy}")
    
    if verbose:
        print()
        print(f"Generated {num_slices} rows of data")
    
    # Write to CSV file
    if verbose:
        print(f"Writing to CSV file: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    if verbose:
        print(f"Successfully wrote Results.csv file: {output_path}")
        print(f"Total rows: {len(rows)} (including header)")


def main():
    """Main function to execute the manual matrix generation workflow."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle round flag (default: True to enable rounding)
    round_to_int = args.use_round
    
    # Handle verbose flag
    verbose = args.verbose
    
    try:
        print("=== Manual Matrix Generation Workflow ===")
        print()
        
        # Count TIF files in input directory
        if verbose:
            print(f"Scanning input directory: {args.input_dir}")
        num_slices = count_tif_files(args.input_dir, verbose=verbose)
        if verbose:
            print(f"Auto-detected {num_slices} slices from TIF files")
            print()
        
        # Generate Results.csv file
        generate_results_csv(
            output_path=args.output,
            start_slice=args.start_slice,
            num_slices=num_slices,
            dx_func=args.dx_func,
            dy_func=args.dy_func,
            round_to_int=round_to_int,
            verbose=verbose
        )
        
        print()
        print("=== Manual matrix generation completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

