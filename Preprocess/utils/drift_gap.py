#!/usr/bin/env python3
"""
Drift Gap Correction Script
This script applies drift corrections from Results.csv to a TIF file
and saves the corrected result as a new TIF file.
"""

import argparse
import os
import glob
import numpy as np
import tifffile
import csv
from pathlib import Path
import natsort
from typing import List, Tuple, Union
from scipy.ndimage import affine_transform
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply drift corrections from Results.csv to TIF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single 3D TIF file
  python drift_gap.py -i /path/to/3d_image.tif -o /path/to/output.tif -t /path/to/Results.csv
  
  # Process directory of 2D TIF files (stack them and apply drift correction)
  python drift_gap.py -i /path/to/2d_images/ -o /path/to/output/ -t /path/to/Results.csv
  
  # Process single 3D file, save corrected slices to directory
  python drift_gap.py -i /path/to/3d_image.tif -o /path/to/output/ -t /path/to/Results.csv
  
  # Using multiple CSV files with custom pattern
  python drift_gap.py -i /path/to/images/ -o /path/to/output/ -t file1.csv file2.csv --pattern "*.tif"
  
  # Verbose output for debugging
  python drift_gap.py -i /path/to/images/ -o /path/to/output/ -t /path/to/Results.csv --verbose
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='/Volumes/T7/20251010_projection/src5/5_reconstruct/reAlign_r_s/part4_o',
        help='Input TIF file path or directory containing TIF files'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/Volumes/T7/20251010_projection/src5/5_reconstruct/reAlign_r_s/part4_o_gap',
        help='Output TIF file path or directory for saving processed files'
    )
    
    parser.add_argument(
        '--results-csv', '-t',
        type=str,
        nargs='+',  # Allow multiple values
        default=[
            '/Volumes/T7/20251010_projection/src5/5_reconstruct/reAlign_r_s/gap1_1to2/gap1_Results.csv',
            '/Volumes/T7/20251010_projection/src5/5_reconstruct/reAlign_r_s/gap2_2to3/Results.csv',
            '/Volumes/T7/20251010_projection/src5/5_reconstruct/reAlign_r_s/gap3_3to4/Results.csv'
            ],
        help='Path(s) to Results.csv file(s) with drift corrections. Multiple files will have their dX, dY values summed.'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.tif',
        help='File pattern to match TIF files (default: *.tif)'
    )
    
    parser.add_argument(
        '--bit-depth',
        choices=['8', '16'],
        default='8',
        help='Output bit depth: 8 or 16 bits (default: 8)'
    )
    
    parser.add_argument(
        '--lzw-compression',
        action='store_true',
        default=True,
        help='Use LZW compression for output files (default: True)'
    )
    
    parser.add_argument(
        '--no-lzw-compression',
        action='store_true',
        help='Disable LZW compression for output files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )
    
    return parser.parse_args()


def load_results_csv(file_paths: Union[str, List[str]], verbose: bool = False) -> List[Tuple[int, float, float]]:
    """
    Load Results.csv file(s) and return list of (slice, dx, dy) tuples.
    If multiple files are provided, their dX, dY values will be summed for each slice.
    
    Args:
        file_paths (Union[str, List[str]]): Path(s) to Results.csv file(s)
        verbose (bool): Enable verbose output
    
    Returns:
        List[Tuple[int, float, float]]: List of (slice, dx, dy) transformations
    """
    # Convert single file path to list for uniform handling
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    # Dictionary to accumulate transformations by slice number
    slice_transformations = {}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results.csv file not found: {file_path}")
        
        if verbose:
            print(f"Loading Results.csv from: {file_path}")
        
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Try DictReader first (header-based)
            try:
                dict_reader = csv.DictReader(csvfile)
                rows = list(dict_reader)
            except Exception:
                rows = []

            # If header-based parsing failed or yielded no rows, fallback to position-based reader
            if not rows:
                csvfile.seek(0)
                raw_reader = csv.reader(csvfile)
                raw_rows = list(raw_reader)
                if not raw_rows:
                    continue
                # Assume first row is header, remaining are data
                header = raw_rows[0]
                data_rows = raw_rows[1:]
                if verbose:
                    print(f"Processing {len(data_rows)} rows from CSV file (position-based)...")
                for row in tqdm(data_rows, desc=f"Loading CSV data from {os.path.basename(file_path)}", disable=not verbose):
                    try:
                        # Try to take last three columns as Slice, dX, dY
                        if len(row) >= 3:
                            slice_num = int(row[-3])
                            dx = float(row[-2])
                            dy = float(row[-1])
                            
                            # Accumulate transformations by slice number
                            if slice_num in slice_transformations:
                                slice_transformations[slice_num] = (
                                    slice_transformations[slice_num][0] + dx,
                                    slice_transformations[slice_num][1] + dy
                                )
                            else:
                                slice_transformations[slice_num] = (dx, dy)
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Skipping invalid row: {row}, error: {e}")
                        continue
            else:
                if verbose:
                    print(f"Processing {len(rows)} rows from CSV file (header-based)...")
                for row in tqdm(rows, desc=f"Loading CSV data from {os.path.basename(file_path)}", disable=not verbose):
                    try:
                        # Some CSVs may contain an extra unnamed first column (e.g., Excel index). Ignore it.
                        # Prefer named keys; fallback to positional if missing.
                        if 'Slice' in row and 'dX' in row and 'dY' in row:
                            slice_num = int(row['Slice'])
                            dx = float(row['dX'])
                            dy = float(row['dY'])
                        else:
                            # Fallback to positional using the values from the row dict in order
                            values = list(row.values())
                            # Take last three as Slice, dX, dY
                            if len(values) >= 3:
                                slice_num = int(values[-3])
                                dx = float(values[-2])
                                dy = float(values[-1])
                            else:
                                continue
                        
                        # Accumulate transformations by slice number
                        if slice_num in slice_transformations:
                            slice_transformations[slice_num] = (
                                slice_transformations[slice_num][0] + dx,
                                slice_transformations[slice_num][1] + dy
                            )
                        else:
                            slice_transformations[slice_num] = (dx, dy)
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Skipping invalid row: {row}, error: {e}")
                        continue
    
    # Convert dictionary back to list of tuples
    transformations = [(slice_num, dx, dy) for slice_num, (dx, dy) in slice_transformations.items()]
    transformations.sort(key=lambda x: x[0])  # Sort by slice number
    
    if verbose:
        print(f"Loaded {len(transformations)} transformations from {len(file_paths)} CSV file(s)")
        if transformations:
            print(f"Slice range: {min(t[0] for t in transformations)} to {max(t[0] for t in transformations)}")
            if len(file_paths) > 1:
                print("Transformations represent summed dX, dY values from all CSV files")
        else:
            print("No valid transformations found!")
    
    return transformations


def is_file_or_directory(path: str) -> str:
    """
    Determine if the given path is a file or directory.
    
    Args:
        path (str): Path to check
    
    Returns:
        str: 'file' if it's a file, 'directory' if it's a directory, 'not_found' if neither
    """
    if not os.path.exists(path):
        return 'not_found'
    elif os.path.isfile(path):
        return 'file'
    elif os.path.isdir(path):
        return 'directory'
    else:
        return 'not_found'


def read_tif_files(input_dir: str, pattern: str = '*.tif', verbose: bool = False) -> Tuple[List[str], List[np.ndarray]]:
    """
    Read all TIF files from the specified directory and sort them naturally.
    
    Args:
        input_dir (str): Path to directory containing TIF files
        pattern (str): File pattern to match (default: '*.tif')
        verbose (bool): Enable verbose output
    
    Returns:
        Tuple[List[str], List[np.ndarray]]: (file_paths, image_data_list)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find all TIF files matching the pattern
    search_pattern = os.path.join(input_dir, pattern)
    tif_files = glob.glob(search_pattern)
    
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {input_dir} matching pattern '{pattern}'")
    
    if verbose:
        print(f"Found {len(tif_files)} TIF files in {input_dir}")
    
    # Sort files naturally using natsort
    sorted_files = natsort.natsorted(tif_files)
    
    if verbose:
        print("Files sorted naturally:")
        for i, file_path in enumerate(sorted_files[:5]):
            print(f"  {i+1}: {os.path.basename(file_path)}")
        if len(sorted_files) > 5:
            print(f"  ... and {len(sorted_files) - 5} more files")
    
    # Read all TIF files
    image_data_list = []
    for i, file_path in enumerate(sorted_files):
        try:
            if verbose and i % 50 == 0:
                print(f"Reading file {i+1}/{len(sorted_files)}: {os.path.basename(file_path)}")
            
            image_data = tifffile.imread(file_path)
            image_data_list.append(image_data)
            
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue
    
    if verbose:
        print(f"Successfully loaded {len(image_data_list)} TIF files")
        if image_data_list:
            print(f"Image shape: {image_data_list[0].shape}")
            print(f"Image dtype: {image_data_list[0].dtype}")
    
    return sorted_files, image_data_list


def read_single_tif_file(file_path: str, verbose: bool = False) -> np.ndarray:
    """
    Read a single TIF file.
    
    Args:
        file_path (str): Path to the TIF file
        verbose (bool): Enable verbose output
    
    Returns:
        np.ndarray: Image data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TIF file not found: {file_path}")
    
    if verbose:
        print(f"Reading TIF file: {file_path}")
    
    try:
        image_data = tifffile.imread(file_path)
        if verbose:
            print(f"Successfully loaded image with shape: {image_data.shape}, dtype: {image_data.dtype}")
        return image_data
    except Exception as e:
        raise RuntimeError(f"Failed to read TIF file {file_path}: {e}")


def apply_drift_correction(image: np.ndarray, dx: float, dy: float, 
                          fill_value: Union[int, float] = 0) -> np.ndarray:
    """
    Apply drift correction to a single image using translation.
    
    Args:
        image (np.ndarray): Input image
        dx (float): Translation in x direction (pixels)
        dy (float): Translation in y direction (pixels)
        fill_value (Union[int, float]): Value to fill empty areas (default: 0 for black)
    
    Returns:
        np.ndarray: Corrected image
    """
    if dx == 0 and dy == 0:
        return image.copy()
    
    # Create transformation matrix
    # For 2D images, we need a 2x2 matrix with translation
    transform_matrix = np.array([[1, 0], [0, 1]])
    offset = np.array([dy, dx])  # Note: scipy uses (y, x) order
    
    # Apply transformation using scipy
    corrected_image = affine_transform(
        image, 
        transform_matrix, 
        offset=offset,
        order=1,  # Linear interpolation
        mode='constant',
        cval=fill_value
    )
    
    return corrected_image


def apply_drift_corrections_to_slices(image_data: np.ndarray, 
                                     transformations: List[Tuple[int, float, float]], 
                                     verbose: bool = False) -> List[np.ndarray]:
    """
    Apply drift corrections to image slices.
    
    Args:
        image_data (np.ndarray): Input image data
        transformations (List[Tuple[int, float, float]]): List of (slice, dx, dy)
        verbose (bool): Enable verbose output
    
    Returns:
        List[np.ndarray]: List of corrected image slices
    """
    if len(image_data.shape) != 3:
        raise ValueError(f"Expected 3D image data, got shape: {image_data.shape}")
    
    num_slices = image_data.shape[0]
    corrected_slices = []
    
    # Detect if CSV specifies a single global shift for all slices
    global_shift: Tuple[float, float] = None  # type: ignore
    if len(transformations) == 1:
        only_slice, only_dx, only_dy = transformations[0]
        # For drift gap correction, if there's only one transformation,
        # treat it as a global shift (apply to all slices)
        global_shift = (only_dx, only_dy)
        if verbose:
            print(f"Detected single transformation: treating as global shift dx={only_dx:.3f}, dy={only_dy:.3f}")

    # Create a mapping from slice number to transformation for per-slice shifts
    transform_map = {slice_num: (dx, dy) for slice_num, dx, dy in transformations}
    
    if verbose:
        print(f"Processing {num_slices} slices with {len(transformations)} transformations")
    
    for i in tqdm(range(num_slices), desc="Applying drift corrections to slices", disable=not verbose):
        slice_num = i + 1  # Assuming slices are 1-indexed
        
        if global_shift is not None:
            dx, dy = global_shift
            if verbose :
                print(f"Processing slice {slice_num}: applying global shift dx={dx:.3f}, dy={dy:.3f}")
        elif slice_num in transform_map:
            dx, dy = transform_map[slice_num]
            if verbose :
                print(f"Processing slice {slice_num}: dx={dx:.3f}, dy={dy:.3f}")
        else:
            dx, dy = 0.0, 0.0
            if verbose :
                print(f"Processing slice {slice_num}: no transformation (dx=0, dy=0)")
        
        # Apply transformation to the slice
        slice_data = image_data[i]
        corrected_slice = apply_drift_correction(slice_data, -dx, -dy)
        corrected_slices.append(corrected_slice)
    
    return corrected_slices


def process_single_file(input_path: str, output_path: str, results_csv_paths: List[str], 
                       args, use_lzw: bool, verbose: bool, bit_depth: int):
    """Process a single TIF file."""
    try:
        print(f"Processing single file: {input_path}")
        
        # Step 1: Read the original TIF file
        print("Step 1: Reading original TIF file...")
        image_data = read_single_tif_file(input_path, verbose=verbose)
        print(f"Successfully loaded image with shape: {image_data.shape}")
        print()
        
        # Step 2: Load Results.csv file(s)
        print("Step 2: Loading Results.csv file(s)...")
        transformations = load_results_csv(results_csv_paths, verbose=verbose)
        print(f"Loaded {len(transformations)} transformations from {len(results_csv_paths)} CSV file(s)")
        print()
        
        # Step 3: Apply drift corrections to slices
        print("Step 3: Applying drift corrections to image slices...")
        corrected_slices = apply_drift_corrections_to_slices(
            image_data, 
            transformations, 
            verbose=verbose
        )
        print(f"Successfully corrected {len(corrected_slices)} slices")
        print()
        
        # Step 4: Save final corrected TIF file
        print("Step 4: Saving final corrected TIF file...")
        save_corrected_tif(
            corrected_slices, 
            output_path, 
            bit_depth=bit_depth, 
            use_lzw=use_lzw, 
            verbose=verbose
        )
        print(f"Successfully saved corrected TIF file to: {output_path}")
        print()
        
        # Save metadata
        metadata_file = os.path.join(os.path.dirname(output_path), 'drift_gap_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("=== SEM Image Drift Gap Correction Metadata ===\n")
            f.write(f"Original TIF file: {input_path}\n")
            f.write(f"Results CSV file(s): {results_csv_paths}\n")
            f.write(f"Output TIF file: {output_path}\n")
            f.write(f"Original image shape: {image_data.shape}\n")
            f.write(f"Number of slices: {len(corrected_slices)}\n")
            f.write(f"Image dtype: {image_data.dtype}\n")
            f.write(f"Output bit depth: {bit_depth} bits\n")
            f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
            f.write(f"Number of transformations: {len(transformations)}\n")
            if transformations:
                f.write(f"Slice range: {min(t[0] for t in transformations)} to {max(t[0] for t in transformations)}\n")
            if len(results_csv_paths) > 1:
                f.write(f"Note: Transformations represent summed dX, dY values from {len(results_csv_paths)} CSV files\n")
        
        print(f"Metadata saved to: {metadata_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir: str, output_dir: str, results_csv_paths: List[str], 
                     args, use_lzw: bool, verbose: bool, bit_depth: int):
    """Process all TIF files in a directory by stacking them and applying drift correction."""
    try:
        print(f"Processing directory: {input_dir}")
        
        # Read all TIF files from directory
        print("Step 1: Reading TIF files from directory...")
        file_paths, image_data_list = read_tif_files(input_dir, args.pattern, verbose=verbose)
        print(f"Found {len(file_paths)} TIF files")
        
        if not image_data_list:
            raise ValueError("No valid TIF files found in directory")
        
        # Check if all images have the same shape
        first_shape = image_data_list[0].shape
        for i, img in enumerate(image_data_list):
            if img.shape != first_shape:
                print(f"Warning: Image {i} has shape {img.shape}, expected {first_shape}")
        
        print(f"All images have shape: {first_shape}")
        print()
        
        # Stack 2D images into 3D array
        print("Step 2: Stacking 2D images into 3D array...")
        image_stack = np.stack(image_data_list, axis=0)
        print(f"Created 3D stack with shape: {image_stack.shape}")
        print()
        
        # Load transformations
        print("Step 3: Loading Results.csv file(s)...")
        transformations = load_results_csv(results_csv_paths, verbose=verbose)
        print(f"Loaded {len(transformations)} transformations from {len(results_csv_paths)} CSV file(s)")
        print()
        
        # Apply drift corrections to the 3D stack
        print("Step 4: Applying drift corrections to 3D stack...")
        corrected_slices = apply_drift_corrections_to_slices(
            image_stack, 
            transformations, 
            verbose=verbose
        )
        print(f"Successfully corrected {len(corrected_slices)} slices")
        print()
        
        # Save each slice as individual 2D TIF file
        print("Step 5: Saving corrected slices as individual 2D TIF files...")
        successful_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine compression
        compression = 'lzw' if use_lzw else None
        
        for i, slice_data in enumerate(corrected_slices):
            try:
                # Generate output filename based on original file order
                if i < len(file_paths):
                    original_name = os.path.splitext(os.path.basename(file_paths[i]))[0]
                    output_filename = f"{original_name}_corrected.tif"
                else:
                    output_filename = f"slice_{i+1:04d}_corrected.tif"
                
                output_file_path = os.path.join(output_dir, output_filename)
                
                if verbose and i % 50 == 0:
                    print(f"Saving slice {i+1}/{len(corrected_slices)}: {output_filename}")
                
                # Convert to appropriate data type for saving
                if slice_data.dtype == np.float32 or slice_data.dtype == np.float64:
                    if bit_depth == 8:
                        slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)
                    else:  # bit_depth == 16
                        slice_data = np.clip(slice_data, 0, 65535).astype(np.uint16)
                elif slice_data.dtype in [np.uint8, np.uint16]:
                    if bit_depth == 8 and slice_data.dtype == np.uint16:
                        slice_data = (slice_data / 256).astype(np.uint8)
                    elif bit_depth == 16 and slice_data.dtype == np.uint8:
                        slice_data = (slice_data.astype(np.uint16) * 256)
                else:
                    if bit_depth == 8:
                        slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)
                    else:  # bit_depth == 16
                        slice_data = np.clip(slice_data, 0, 65535).astype(np.uint16)
                
                # Save using tifffile
                tifffile.imwrite(
                    output_file_path,
                    slice_data,
                    compression=compression,
                    metadata={'description': f'Drift corrected slice {i+1} from {input_dir}'}
                )
                
                successful_count += 1
                
            except Exception as e:
                print(f"Error saving slice {i+1}: {e}")
                continue
        
        print(f"Successfully saved {successful_count}/{len(corrected_slices)} corrected slices")
        print()
        
        # Save batch metadata
        metadata_file = os.path.join(output_dir, 'batch_drift_gap_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("=== Batch SEM Image Drift Gap Correction Metadata ===\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Results CSV file(s): {results_csv_paths}\n")
            f.write(f"Pattern: {args.pattern}\n")
            f.write(f"Total files found: {len(file_paths)}\n")
            f.write(f"Total slices processed: {len(corrected_slices)}\n")
            f.write(f"Successfully saved slices: {successful_count}\n")
            f.write(f"Original image shape: {first_shape}\n")
            f.write(f"3D stack shape: {image_stack.shape}\n")
            f.write(f"Output bit depth: {bit_depth} bits\n")
            f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
            f.write(f"Number of transformations: {len(transformations)}\n")
            if transformations:
                f.write(f"Slice range: {min(t[0] for t in transformations)} to {max(t[0] for t in transformations)}\n")
            if len(results_csv_paths) > 1:
                f.write(f"Note: Transformations represent summed dX, dY values from {len(results_csv_paths)} CSV files\n")
        
        print(f"Batch metadata saved to: {metadata_file}")
        return True
        
    except Exception as e:
        print(f"Error processing directory {input_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_corrected_tif(corrected_slices: List[np.ndarray], 
                      output_path: str, 
                      bit_depth: int = 8, 
                      use_lzw: bool = True, 
                      verbose: bool = False):
    """
    Save corrected slices as a single TIF file.
    
    Args:
        corrected_slices (List[np.ndarray]): List of corrected image slices
        output_path (str): Output TIF file path
        bit_depth (int): Output bit depth (8 or 16)
        use_lzw (bool): Whether to use LZW compression
        verbose (bool): Enable verbose output
    """
    if not corrected_slices:
        raise ValueError("No corrected slices to save")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if verbose:
        print(f"Saving corrected TIF file to: {output_path}")
        print(f"Number of slices: {len(corrected_slices)}")
        print(f"Slice shape: {corrected_slices[0].shape}")
    
    # Stack slices into 3D array
    corrected_array = np.stack(corrected_slices, axis=0)
    
    if verbose:
        print(f"Final array shape: {corrected_array.shape}")
    
    # Convert to appropriate data type
    if verbose:
        print("Converting data type...")
    
    if corrected_array.dtype == np.float32 or corrected_array.dtype == np.float64:
        if bit_depth == 8:
            corrected_array = np.clip(corrected_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            corrected_array = np.clip(corrected_array, 0, 65535).astype(np.uint16)
    elif corrected_array.dtype in [np.uint8, np.uint16]:
        if bit_depth == 8 and corrected_array.dtype == np.uint16:
            corrected_array = (corrected_array / 256).astype(np.uint8)
        elif bit_depth == 16 and corrected_array.dtype == np.uint8:
            corrected_array = (corrected_array.astype(np.uint16) * 256)
    else:
        if bit_depth == 8:
            corrected_array = np.clip(corrected_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            corrected_array = np.clip(corrected_array, 0, 65535).astype(np.uint16)
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    # Save using tifffile with progress bar
    if verbose:
        print("Saving TIF file...")
    
    # Save using tifffile
    tifffile.imwrite(
        output_path,
        corrected_array,
        compression=compression,
        metadata={'description': 'SEM images with drift gap correction'}
    )
    
    if verbose:
        print(f"Successfully saved corrected TIF file: {output_path}")


def main():
    """Main function to execute the drift gap correction workflow."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression and not args.no_lzw_compression
    
    # Handle verbose flag
    verbose = args.verbose 
    
    # Convert bit depth to integer
    bit_depth = int(args.bit_depth)
    
    # Define file paths based on command line arguments
    input_path = args.input_dir
    output_path = args.output_dir
    results_csv_paths = args.results_csv
    
    try:
        print("=== SEM Image Drift Gap Correction Workflow ===")
        print(f"Output bit depth: {bit_depth} bits")
        print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
        print()
        
        # Determine input type
        input_type = is_file_or_directory(input_path)
        if input_type == 'not_found':
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        # Determine output type
        output_type = is_file_or_directory(output_path)
        if output_type == 'not_found':
            # If output doesn't exist, determine if it should be a file or directory
            # based on input type
            if input_type == 'file':
                output_type = 'file'
            else:
                output_type = 'directory'
        
        print(f"Input type: {input_type}")
        print(f"Output type: {output_type}")
        print()
        
        # Process based on input type
        if input_type == 'file':
            # Single file processing
            if output_type == 'directory':
                # If output is directory, create filename based on input
                input_name = os.path.splitext(os.path.basename(input_path))[0]
                output_file_path = os.path.join(output_path, f"{input_name}_corrected.tif")
            else:
                output_file_path = output_path
            
            success = process_single_file(
                input_path, output_file_path, results_csv_paths,
                args, use_lzw, verbose, bit_depth
            )
            
        else:  # input_type == 'directory'
            # Directory processing
            if output_type == 'file':
                raise ValueError("Cannot process directory input with file output. Please specify a directory for output.")
            
            success = process_directory(
                input_path, output_path, results_csv_paths,
                args, use_lzw, verbose, bit_depth
            )
        
        if success:
            print("=== Drift gap correction workflow completed successfully! ===")
        else:
            print("=== Drift gap correction workflow completed with errors! ===")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
