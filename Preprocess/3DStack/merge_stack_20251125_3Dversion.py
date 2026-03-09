#!/usr/bin/env python3
"""
Drift Part Image Stack Merging Script
This script reads specific drift_part TIF files (drift_part1.tif through drift_part4.tif)
and merges them into a single stacked TIF file by concatenating along the z-axis.
The script expects images with shapes:
- drift_part1.tif: (231, 4819, 3424)
- drift_part2.tif: (168, 4819, 3424) 
- drift_part3.tif: (246, 4819, 3424)
- drift_part4.tif: (195, 4819, 3424)

The output will have shape: (840, 4819, 3424) where 840 = 231+168+246+195
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
        description="Merge drift_part TIF files into a single stacked TIF file along z-axis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python merge_stack.py -i /path/to/input/dir -o /path/to/output.tif
  
  # With 16-bit output and no compression
  python merge_stack.py -i /path/to/input/dir -o /path/to/output.tif --bit-depth 16 --no-lzw-compression
  
  # Verbose output
  python merge_stack.py -i /path/to/input/dir -o /path/to/output.tif --verbose
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
        '--output-file', '-o',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/4_merge/drift_merged.tif',
        # required=True,
        help='Output TIF file path for the merged drift stack'
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
    
    return parser.parse_args()


def read_drift_part_files(input_dir: str, verbose: bool = False) -> Tuple[List[str], List[np.ndarray]]:
    """
    Read every TIF file inside input_dir (natsort order) and return per-file arrays.
    Each file can either be a full stack shaped (slices, width, height) or a single slice
    shaped (width, height). All data are aligned along the slice dimension.
    
    Args:
        input_dir (str): Path to directory containing TIF files
        verbose (bool): Enable verbose output
    
    Returns:
        Tuple[List[str], List[np.ndarray]]: (file_paths, image_data_list)
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
    
    file_paths: List[str] = []
    image_data_list: List[np.ndarray] = []
    base_xy: Optional[Tuple[int, int]] = None
    
    if verbose:
        print(f"Reading {len(tif_paths)} TIF files from {input_dir} (natsorted)")
    
    for idx, tif_path in enumerate(tif_paths, 1):
        try:
            image_data = tifffile.imread(tif_path)
        except Exception as e:
            print(f"Error: Failed to read {tif_path}: {e}")
            raise
        
        if image_data.ndim == 2:
            image_data = np.expand_dims(image_data, axis=0)
        elif image_data.ndim == 3:
            pass
        else:
            raise ValueError(
                f"Unsupported image dimensions for {tif_path.name}: {image_data.shape}"
            )
        
        if base_xy is None:
            base_xy = image_data.shape[1:]
        elif image_data.shape[1:] != base_xy:
            raise ValueError(
                f"XY dimensions mismatch for {tif_path.name}: {image_data.shape[1:]} "
                f"(expected {base_xy})"
            )
        
        file_paths.append(str(tif_path))
        image_data_list.append(image_data)
        
        if verbose:
            rel_path = tif_path.relative_to(input_path)
            print(f"  File {idx}/{len(tif_paths)}: {rel_path}")
            print(f"    Shape (after normalization): {image_data.shape}")
            print(f"    Dtype: {image_data.dtype}")
    
    if verbose:
        print(f"Successfully loaded {len(image_data_list)} TIF files with XY={base_xy}")
    
    return file_paths, image_data_list


def merge_images(image_data_list: List[np.ndarray], verbose: bool = False) -> np.ndarray:
    """
    Merge a list of images into a single stacked array by concatenating along z-axis.
    
    Args:
        image_data_list (List[np.ndarray]): List of image arrays
        verbose (bool): Enable verbose output
    
    Returns:
        np.ndarray: Stacked image array
    """
    if not image_data_list:
        raise ValueError("No images to merge")
    
    if verbose:
        print(f"Merging {len(image_data_list)} images along z-axis...")
    
    # Check if all images have compatible shapes for z-axis concatenation
    # All images should have the same x,y dimensions but can have different z dimensions
    first_shape = image_data_list[0].shape
    if verbose:
        print(f"First image shape: {first_shape}")
    
    # Verify that all images have the same x,y dimensions
    for i, img in enumerate(image_data_list[1:], 1):
        if len(img.shape) != len(first_shape):
            raise ValueError(f"Image {i} has {len(img.shape)} dimensions, expected {len(first_shape)}")
        
        # Check x,y dimensions (all dimensions except the first one)
        if img.shape[1:] != first_shape[1:]:
            raise ValueError(f"Image {i} has shape {img.shape}, x,y dimensions {img.shape[1:]} don't match expected {first_shape[1:]}")
        
        if verbose:
            print(f"Image {i+1} shape: {img.shape}")
    
    # Concatenate images along the z-axis (axis=0)
    merged_array = np.concatenate(image_data_list, axis=0)
    
    if verbose:
        print(f"Merged array shape: {merged_array.shape}")
        print(f"Merged array dtype: {merged_array.dtype}")
        print(f"Total z-slices: {merged_array.shape[0]}")
        print(f"XY dimensions: {merged_array.shape[1:]}")
    
    return merged_array


def convert_bit_depth(image_array: np.ndarray, bit_depth: int, verbose: bool = False) -> np.ndarray:
    """
    Convert image array to specified bit depth.
    
    Args:
        image_array (np.ndarray): Input image array
        bit_depth (int): Target bit depth (8 or 16)
        verbose (bool): Enable verbose output
    
    Returns:
        np.ndarray: Converted image array
    """
    if verbose:
        print(f"Converting to {bit_depth}-bit depth...")
        print(f"Original dtype: {image_array.dtype}")
    
    # Handle different input data types
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        if bit_depth == 8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            image_array = np.clip(image_array, 0, 65535).astype(np.uint16)
    elif image_array.dtype in [np.uint8, np.uint16]:
        if bit_depth == 8 and image_array.dtype == np.uint16:
            image_array = (image_array / 256).astype(np.uint8)
        elif bit_depth == 16 and image_array.dtype == np.uint8:
            image_array = (image_array.astype(np.uint16) * 256)
    else:
        if bit_depth == 8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            image_array = np.clip(image_array, 0, 65535).astype(np.uint16)
    
    if verbose:
        print(f"Converted dtype: {image_array.dtype}")
    
    return image_array


def save_merged_tif(merged_array: np.ndarray, 
                   output_path: str, 
                   bit_depth: int = 8, 
                   use_lzw: bool = True, 
                   verbose: bool = False):
    """
    Save merged image array as a TIF file.
    
    Args:
        merged_array (np.ndarray): Merged image array
        output_path (str): Output TIF file path
        bit_depth (int): Output bit depth (8 or 16)
        use_lzw (bool): Whether to use LZW compression
        verbose (bool): Enable verbose output
    """
    if verbose:
        print(f"Saving merged TIF file to: {output_path}")
        print(f"Array shape: {merged_array.shape}")
        print(f"Array dtype: {merged_array.dtype}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to appropriate data type
    converted_array = convert_bit_depth(merged_array, bit_depth, verbose)
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    if verbose:
        print(f"Using compression: {compression}")
        print("Saving TIF file...")
    
    # Save using tifffile
    tifffile.imwrite(
        output_path,
        converted_array,
        compression=compression,
        metadata={'description': f'Merged stack from {merged_array.shape[0]} TIF files'}
    )
    
    if verbose:
        print(f"Successfully saved merged TIF file: {output_path}")


def save_metadata(file_paths: List[str], 
                 output_path: str, 
                 merged_shape: Tuple[int, ...], 
                 bit_depth: int, 
                 use_lzw: bool, 
                 verbose: bool = False):
    """
    Save metadata about the merging process.
    
    Args:
        file_paths (List[str]): List of input file paths
        output_path (str): Output file path
        merged_shape (Tuple[int, ...]): Shape of merged array
        bit_depth (int): Output bit depth
        use_lzw (bool): Whether LZW compression was used
        verbose (bool): Enable verbose output
    """
    metadata_file = os.path.join(os.path.dirname(output_path), 'merge_metadata.txt')
    
    if verbose:
        print(f"Saving metadata to: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("=== TIF Stack Merge Metadata ===\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Number of input files: {len(file_paths)}\n")
        f.write(f"Merged array shape: {merged_shape}\n")
        f.write(f"Output bit depth: {bit_depth} bits\n")
        f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
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
        print(f"Output file: {args.output_file}")
        print(f"Output bit depth: {bit_depth} bits")
        print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
        print()
        
        # Step 1: Read drift_part TIF files from input directory
        print("Step 1: Reading drift_part TIF files from input directory...")
        file_paths, image_data_list = read_drift_part_files(
            args.input_dir, 
            verbose=verbose
        )
        print(f"Successfully loaded {len(image_data_list)} drift_part files")
        print()
        
        # Step 2: Merge images into a single stack along z-axis
        print("Step 2: Merging images into a single stack along z-axis...")
        merged_array = merge_images(image_data_list, verbose=verbose)
        print(f"Successfully merged {len(image_data_list)} images")
        print(f"Final stack shape: {merged_array.shape}")
        print()
        
        # Step 3: Save merged TIF file
        print("Step 3: Saving merged TIF file...")
        save_merged_tif(
            merged_array, 
            args.output_file, 
            bit_depth=bit_depth, 
            use_lzw=use_lzw, 
            verbose=verbose
        )
        print(f"Successfully saved merged TIF file to: {args.output_file}")
        print()
        
        # Step 4: Save metadata
        print("Step 4: Saving metadata...")
        save_metadata(
            file_paths, 
            args.output_file, 
            merged_array.shape, 
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
