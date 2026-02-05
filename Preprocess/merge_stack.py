#!/usr/bin/env python3
"""
Drift Part Image Stack Processing Script
This script reads TIF files from a directory and extracts all slices,
saving each slice as an individual 2D TIF file (without stacking into 3D array).
The script can handle both 2D and 3D input images.
"""

import argparse
import os
import numpy as np
import tifffile
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from natsort import natsorted


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract slices from TIF files and save as individual 2D TIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python merge_stack.py -i /path/to/input/dir -o /path/to/output/dir
  
  # With 16-bit output and no compression
  python merge_stack.py -i /path/to/input/dir -o /path/to/output/dir --bit-depth 16 --no-lzw-compression
  
  # Verbose output
  python merge_stack.py -i /path/to/input/dir -o /path/to/output/dir --verbose
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/3_drift',
        # required=True,
        help='Input directory containing drift_part TIF files to merge'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/4_merge',
        # required=True,
        help='Output directory for saving individual slice TIF files'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Legacy alias for --output-dir (kept for backward compatibility)'
    )
    
    parser.add_argument(
        '--bit-depth',
        choices=['8', '16'],
        default='8',
        help='Output bit depth: 8 or 16 bits (default: 8)'
    )
    
    parser.add_argument(
        '--lzw-compression',
        type=bool,
        default=False,
        help='Use LZW compression for output files (default: True, no manual input needed)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=bool,
        default=True,
        help='Enable verbose output. Set to True to enable, False to disable (default: True). Note: This parameter cannot be manually input from command line.'
    )
    
    args = parser.parse_args()
    
    # Backward compatibility: allow scripts still using --output-file
    if args.output_file:
        if args.verbose:
            print(f"[Warning] --output-file 已弃用，将其视为 --output-dir: {args.output_file}")
        args.output_dir = args.output_file
    
    return args


def collect_drift_part_files(input_dir: str, verbose: bool = False) -> List[Path]:
    """
    Collect drift_part TIF file paths (natsort order) without loading them into memory.
    
    Args:
        input_dir (str): Path to directory containing TIF files
        verbose (bool): Enable verbose output
    
    Returns:
        List[Path]: Sorted list of TIF file paths
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    tif_paths = [
        p for p in input_path.rglob('*')
        if p.is_file()
        and not p.name.startswith('.')
        and p.suffix.lower() in ('.tif', '.tiff')
    ]
    
    if not tif_paths:
        raise FileNotFoundError(f"No TIF files found under: {input_dir}")
    
    tif_paths = natsorted(
        tif_paths,
        key=lambda p: str(p.relative_to(input_path))
    )
    
    if verbose:
        print(f"Found {len(tif_paths)} TIF files under {input_dir}")
    
    return tif_paths


def process_and_save_slices(file_paths: List[Path],
                            output_dir: str,
                            bit_depth: int = 8,
                            use_lzw: bool = True,
                            verbose: bool = False) -> Tuple[int, Tuple[int, int]]:
    """
    Stream each drift_part file, extract slices, and save them immediately to disk.
    
    Args:
        file_paths (List[Path]): List of TIF file paths
        output_dir (str): Output directory path
        bit_depth (int): Output bit depth (8 or 16)
        use_lzw (bool): Whether to use LZW compression
        verbose (bool): Enable verbose output
    
    Returns:
        Tuple[int, Tuple[int, int]]: (total_slices, slice_shape)
    """
    if not file_paths:
        raise ValueError("No files provided for processing")
    
    os.makedirs(output_dir, exist_ok=True)
    compression = 'lzw' if use_lzw else None
    
    total_slices = 0
    slice_shape: Optional[Tuple[int, int]] = None
    
    if verbose:
        print(f"Saving slices to: {output_dir}")
        print(f"Compression: {'LZW' if compression else 'None'}")
    
    for idx, tif_path in enumerate(file_paths, 1):
        if verbose:
            rel_path = tif_path
            print(f"  [{idx}/{len(file_paths)}] Loading {rel_path}")
        
        try:
            image_data = tifffile.imread(tif_path)
        except Exception as e:
            print(f"Error: Failed to read {tif_path}: {e}")
            raise
        
        if image_data.ndim == 2:
            image_data = np.expand_dims(image_data, axis=0)
        elif image_data.ndim != 3:
            raise ValueError(
                f"Unsupported image dimensions for {tif_path.name}: {image_data.shape}"
            )
        
        current_xy = image_data.shape[1:]
        if slice_shape is None:
            slice_shape = current_xy
        elif current_xy != slice_shape:
            raise ValueError(
                f"XY dimensions mismatch for {tif_path.name}: {current_xy} "
                f"(expected {slice_shape})"
            )
        
        for z in range(image_data.shape[0]):
            slice_data = image_data[z, :, :]
            converted_slice = convert_bit_depth_slice(slice_data, bit_depth)
            output_filename = f"slice_{total_slices:06d}.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            tifffile.imwrite(
                output_path,
                converted_slice,
                compression=compression,
                metadata={'description': f'Slice {total_slices} from merged stack'}
            )
            
            total_slices += 1
            
            if verbose and total_slices % 100 == 0:
                print(f"    Saved {total_slices} slices so far...")
        
        del image_data
    
    if slice_shape is None:
        raise RuntimeError("Failed to determine slice shape during processing")
    
    if verbose:
        print(f"Finished streaming save. Total slices: {total_slices}")
    
    return total_slices, slice_shape


def convert_bit_depth_slice(slice_array: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Convert a single 2D slice to specified bit depth.
    
    Args:
        slice_array (np.ndarray): Input 2D slice array
        bit_depth (int): Target bit depth (8 or 16)
    
    Returns:
        np.ndarray: Converted slice array
    """
    # Handle different input data types
    if slice_array.dtype == np.float32 or slice_array.dtype == np.float64:
        if bit_depth == 8:
            return np.clip(slice_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            return np.clip(slice_array, 0, 65535).astype(np.uint16)
    elif slice_array.dtype in [np.uint8, np.uint16]:
        if bit_depth == 8 and slice_array.dtype == np.uint16:
            return (slice_array / 256).astype(np.uint8)
        elif bit_depth == 16 and slice_array.dtype == np.uint8:
            return (slice_array.astype(np.uint16) * 256)
        else:
            return slice_array
    else:
        if bit_depth == 8:
            return np.clip(slice_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            return np.clip(slice_array, 0, 65535).astype(np.uint16)


def save_metadata(file_paths: List[str], 
                 output_dir: str, 
                 num_slices: int,
                 slice_shape: Tuple[int, ...], 
                 bit_depth: int, 
                 use_lzw: bool, 
                 verbose: bool = False):
    """
    Save metadata about the merging process.
    
    Args:
        file_paths (List[str]): List of input file paths
        output_dir (str): Output directory path
        num_slices (int): Total number of slices extracted
        slice_shape (Tuple[int, ...]): Shape of individual slices
        bit_depth (int): Output bit depth
        use_lzw (bool): Whether LZW compression was used
        verbose (bool): Enable verbose output
    """
    metadata_file = os.path.join(output_dir, 'merge_metadata.txt')
    
    if verbose:
        print(f"Saving metadata to: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("=== TIF Stack Merge Metadata ===\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Number of input files: {len(file_paths)}\n")
        f.write(f"Total slices extracted: {num_slices}\n")
        f.write(f"Slice shape: {slice_shape}\n")
        f.write(f"Output bit depth: {bit_depth} bits\n")
        f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
        f.write(f"Storage mode: Individual 2D files (not stacked)\n")
        f.write(f"Input files:\n")
        
        for i, file_path in enumerate(file_paths):
            f.write(f"  {i+1:3d}: {os.path.basename(file_path)}\n")
    
    if verbose:
        print(f"Metadata saved to: {metadata_file}")


def main():
    """Main function to execute the merge stack workflow."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression 
    
    # Handle verbose flag
    verbose = args.verbose 
    
    # Convert bit depth to integer
    bit_depth = int(args.bit_depth)
    
    try:
        print("=== Drift Part Stack Merge Workflow ===")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output bit depth: {bit_depth} bits")
        print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
        print()
        
        # Step 1: Collect drift_part TIF file paths
        print("Step 1: Collecting drift_part TIF file paths...")
        file_paths = collect_drift_part_files(
            args.input_dir,
            verbose=verbose
        )
        print(f"Found {len(file_paths)} drift_part files")
        print()
        
        # Step 2: Stream slices and save immediately to disk
        print("Step 2: Streaming slices and saving to disk...")
        total_slices, slice_shape = process_and_save_slices(
            file_paths,
            args.output_dir,
            bit_depth=bit_depth,
            use_lzw=use_lzw,
            verbose=verbose
        )
        print(f"Successfully saved {total_slices} slices to: {args.output_dir}")
        print(f"Slice shape: {slice_shape}")
        print()
        
        # Step 3: Save metadata
        print("Step 3: Saving metadata...")
        save_metadata(
            [str(p) for p in file_paths], 
            args.output_dir, 
            total_slices,
            slice_shape, 
            bit_depth, 
            use_lzw, 
            verbose=verbose
        )
        print("Metadata saved successfully")
        print()
        
        print("=== Merge workflow completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
