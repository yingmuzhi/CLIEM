#!/usr/bin/env python3
"""
Image Drift Correction Script
This script corrects image drift by applying translation matrices to 2D images
and fills empty areas with black pixels.
"""

import argparse
import os
import glob
import numpy as np
import tifffile
import json
import csv
from pathlib import Path
import natsort
from typing import List, Tuple, Union
from scipy.ndimage import affine_transform
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Correct image drift using translation matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single 3D TIF file
  python align_correct_background.py -i /path/to/3d_image.tif -o /path/to/output.tif -t transformations.csv
  
  # Process directory of 2D TIF files (stack them and apply alignment)
  python align_correct_background.py -i /path/to/2d_images/ -o /path/to/output/ -t transformations.csv
  
  # Process single 3D file, save aligned slices to directory
  python align_correct_background.py -i /path/to/3d_image.tif -o /path/to/output/ -t transformations.csv
  
  # Using JSON transformation file with custom pattern
  python align_correct_background.py -i /path/to/images/ -o /path/to/output/ -t transformations.json --pattern "*.tif"
  
  # Verbose output for debugging
  python align_correct_background.py -i /path/to/images/ -o /path/to/output/ -t transformations.csv --verbose
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1_2_pad/part1',
        help='Input TIF file path or directory containing TIF files'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1',
        help='Output TIF file path or directory for saving processed files'
    )
    
    parser.add_argument(
        '--transformations', '-t',
        type=str,
        default='/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1_2_pad/part1/Results.csv',
        help='Path to transformation file (CSV or JSON format)'
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
        type=bool,
        default=True,
        help='Use LZW compression for output files (default: True, no manual input needed)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=bool,
        default=True,
        help='Enable verbose output (default: True)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=5000.0,
        help='Threshold for preprocessing transformations (default: 50.0)'
    )
    
    parser.add_argument(
        '--drift-gap',
        type=bool,
        default=False,
        help='Apply drift from the subsequent slice to the reference slice (default: False)'
    )
    
    return parser.parse_args()


def load_transformations(file_path: str) -> List[Tuple[float, float]]:
    """
    Load transformation matrices from file.
    
    Args:
        file_path (str): Path to transformation file (CSV or JSON)
    
    Returns:
        List[Tuple[float, float]]: List of (dx, dy) translation values
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transformation file not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return load_transformations_csv(file_path)
    elif file_ext == '.json':
        return load_transformations_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .csv, .json")


def load_transformations_csv(file_path: str) -> List[Tuple[float, float]]:
    """Load transformations from CSV file."""
    transformations = []
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        # Try to detect delimiter
        sample = csvfile.read(1024)
        csvfile.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.reader(csvfile, delimiter=delimiter)
        
        # Skip header if present
        first_row = next(reader, None)
        if first_row and not all(s.replace('.', '').replace('-', '').isdigit() for s in first_row[:2]):
            # First row looks like header, process it
            try:
                dx, dy = float(first_row[0]), float(first_row[1])
                transformations.append((dx, dy))
            except (ValueError, IndexError):
                pass  # Skip header row
        else:
            # First row is data
            if first_row:
                try:
                    dx, dy = float(first_row[0]), float(first_row[1])
                    transformations.append((dx, dy))
                except (ValueError, IndexError):
                    pass
        
        # Process remaining rows
        for row in reader:
            if len(row) >= 2:
                try:
                    dx, dy = float(row[0]), float(row[1])
                    transformations.append((dx, dy))
                except ValueError:
                    continue
    
    return transformations


def load_transformations_json(file_path: str) -> List[Tuple[float, float]]:
    """Load transformations from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    transformations = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        # List of [dx, dy] or [{"dx": x, "dy": y}, ...]
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                transformations.append((float(item[0]), float(item[1])))
            elif isinstance(item, dict):
                dx = item.get('dx', item.get('x', item.get('translation_x', 0)))
                dy = item.get('dy', item.get('y', item.get('translation_y', 0)))
                transformations.append((float(dx), float(dy)))
    elif isinstance(data, dict):
        # Dictionary with transformations key
        if 'transformations' in data:
            for item in data['transformations']:
                if isinstance(item, list) and len(item) >= 2:
                    transformations.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    dx = item.get('dx', item.get('x', item.get('translation_x', 0)))
                    dy = item.get('dy', item.get('y', item.get('translation_y', 0)))
                    transformations.append((float(dx), float(dy)))
    
    return transformations


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


def correct_images(image_data_list: List[np.ndarray], 
                  transformations: List[Tuple[float, float]], 
                  verbose: bool = False) -> List[np.ndarray]:
    """
    Apply drift correction to all images.
    
    Args:
        image_data_list (List[np.ndarray]): List of image arrays
        transformations (List[Tuple[float, float]]): List of (dx, dy) transformations
        verbose (bool): Enable verbose output
    
    Returns:
        List[np.ndarray]: List of corrected image arrays
    """
    if not image_data_list:
        return []
    
    if len(transformations) != len(image_data_list):
        print(f"Warning: Number of transformations ({len(transformations)}) doesn't match "
              f"number of images ({len(image_data_list)}). Using available transformations.")
    
    corrected_images = []
    
    for i, (image, transform) in enumerate(zip(image_data_list, transformations)):
        if verbose and i % 50 == 0:
            print(f"Correcting image {i+1}/{len(image_data_list)}")
        
        dx, dy = transform
        corrected_image = apply_drift_correction(image, dx, dy)
        corrected_images.append(corrected_image)
    
    return corrected_images


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


def load_results_csv(file_path: str, verbose: bool = False) -> List[Tuple[int, float, float]]:
    """
    Load Results.csv file and return list of (slice, dx, dy) tuples.
    
    Args:
        file_path (str): Path to Results.csv file
        verbose (bool): Enable verbose output
    
    Returns:
        List[Tuple[int, float, float]]: List of (slice, dx, dy) transformations
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results.csv file not found: {file_path}")
    
    if verbose:
        print(f"Loading Results.csv from: {file_path}")
    
    transformations = []
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # Convert to list to get total count
        
        if verbose:
            print(f"Processing {len(rows)} rows from CSV file...")
        
        for row in tqdm(rows, desc="Loading CSV data", disable=not verbose):
            try:
                slice_num = int(row['Slice'])
                dx = float(row['dX'])
                dy = float(row['dY'])
                transformations.append((slice_num, dx, dy))
            except (ValueError, KeyError) as e:
                if verbose:
                    print(f"Warning: Skipping invalid row: {row}, error: {e}")
                continue
    
    if verbose:
        print(f"Loaded {len(transformations)} transformations from Results.csv")
        if transformations:
            print(f"Slice range: {min(t[0] for t in transformations)} to {max(t[0] for t in transformations)}")
        else:
            print("No valid transformations found!")
    
    return transformations


def preprocess_transformations(transformations: List[Tuple[int, float, float]], 
                              threshold: float, 
                              apply_drift_gap: bool = False,
                              verbose: bool = False) -> List[Tuple[int, float, float]]:
    """
    Preprocess transformations by applying threshold-based correction.
    
    Algorithm:
    1. First record the reference slice number
    2. Traverse from the beginning according to Results.csv index
    3. If the absolute difference between adjacent slices exceeds threshold 
       and is not adjacent to the reference slice, replace with previous slice's dX, dY
    
    Args:
        transformations (List[Tuple[int, float, float]]): List of (slice, dx, dy)
        threshold (float): Threshold for difference detection
        apply_drift_gap (bool): Whether to apply drift gap to reference slice
        verbose (bool): Enable verbose output
    
    Returns:
        List[Tuple[int, float, float]]: Preprocessed transformations
    """
    if not transformations:
        return []
    
    # Step 1: Find the reference slice (the one not in the list, with dx=0, dy=0)
    all_slices = set(t[0] for t in transformations)
    max_slice = max(all_slices)
    reference_slice = None
    
    # Find the missing slice (reference)
    for i in range(1, max_slice + 1):
        if i not in all_slices:
            reference_slice = i
            break
    
    if reference_slice is None:
        # If no missing slice found, use slice 1 as reference
        reference_slice = 1
        if verbose:
            print("Warning: No reference slice found, using slice 1 as reference")
    
    if verbose:
        print(f"Reference slice: {reference_slice}")
    
    # Step 2: Process transformations according to Results.csv index order (original order)
    preprocessed = []
    prev_dx, prev_dy = 0.0, 0.0
    
    for i, (slice_num, dx, dy) in enumerate(tqdm(transformations, desc="Preprocessing transformations", disable=not verbose)):
        # Calculate difference from previous transformation
        dx_diff = abs(dx - prev_dx)
        dy_diff = abs(dy - prev_dy)
        max_diff = max(dx_diff, dy_diff)
        
        # Check if current slice is adjacent to reference slice
        is_adjacent_to_reference = abs(slice_num - reference_slice) == 1
        
        # Apply correction logic
        if max_diff > threshold and not is_adjacent_to_reference:
            if verbose:
                print(f"Slice {slice_num}: Difference {max_diff:.2f} > threshold {threshold} "
                      f"and not adjacent to reference slice {reference_slice}, using previous values")
            # Use previous transformation values
            corrected_dx, corrected_dy = prev_dx, prev_dy
        else:
            if verbose and max_diff > threshold:
                print(f"Slice {slice_num}: Difference {max_diff:.2f} > threshold {threshold} "
                      f"but adjacent to reference slice {reference_slice}, keeping original values")
            corrected_dx, corrected_dy = dx, dy
        
        preprocessed.append((slice_num, corrected_dx, corrected_dy))
        prev_dx, prev_dy = corrected_dx, corrected_dy
    
    # Determine reference slice transformation
    reference_dx, reference_dy = 0.0, 0.0
    if apply_drift_gap:
        transform_map = {slice_num: (dx, dy) for slice_num, dx, dy in preprocessed}
        next_slice = reference_slice + 1
        if next_slice in transform_map:
            reference_dx, reference_dy = transform_map[next_slice]
            if verbose:
                print(f"Reference slice {reference_slice}: applying drift gap using slice "
                      f"{next_slice} (dx={reference_dx:.3f}, dy={reference_dy:.3f})")
        else:
            if verbose:
                print(f"Reference slice {reference_slice}: drift gap enabled but no following slice "
                      f"found, using default dx=0, dy=0")
    
    # Add reference slice (shifted or zero) at the end
    preprocessed.append((reference_slice, reference_dx, reference_dy))
    
    if verbose:
        print(f"Preprocessed {len(preprocessed)} transformations")
        print(f"Reference slice: {reference_slice}")
        print(f"Adjacent slices to reference: {reference_slice-1}, {reference_slice+1}")
    
    return preprocessed


def save_corrected_results_csv(transformations: List[Tuple[int, float, float]], 
                              output_path: str, 
                              verbose: bool = False):
    """
    Save corrected transformations to Results_changed.csv.
    
    Args:
        transformations (List[Tuple[int, float, float]]): List of (slice, dx, dy)
        output_path (str): Output CSV file path
        verbose (bool): Enable verbose output
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if verbose:
        print(f"Saving corrected Results.csv to: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['slice', 'dx', 'dy'])  # Header
        
        for slice_num, dx, dy in tqdm(transformations, desc="Saving corrected CSV", disable=not verbose):
            writer.writerow([slice_num, dx, dy])
    
    if verbose:
        print(f"Successfully saved {len(transformations)} transformations to Results_changed.csv")


def apply_alignment_to_slices(image_data, 
                             transformations: List[Tuple[int, float, float]], 
                             verbose: bool = False) -> List[np.ndarray]:
    """
    Apply alignment transformations to image slices.
    
    Args:
        image_data: Input image data - can be either:
            - 3D np.ndarray (shape: [num_slices, height, width])
            - List[np.ndarray] of 2D images
        transformations (List[Tuple[int, float, float]]): List of (slice, dx, dy)
        verbose (bool): Enable verbose output
    
    Returns:
        List[np.ndarray]: List of aligned image slices
    """
    # Handle both 3D array and list of 2D arrays
    if isinstance(image_data, list):
        # Direct processing of 2D image list (memory efficient)
        image_list = image_data
        num_slices = len(image_list)
    elif isinstance(image_data, np.ndarray):
        if len(image_data.shape) == 3:
            # 3D array: convert to list for processing
            num_slices = image_data.shape[0]
            image_list = [image_data[i] for i in range(num_slices)]
        elif len(image_data.shape) == 2:
            # Single 2D image
            image_list = [image_data]
            num_slices = 1
        else:
            raise ValueError(f"Unsupported image data shape: {image_data.shape}")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")
    
    aligned_slices = []
    
    # Create a mapping from slice number to transformation
    transform_map = {slice_num: (dx, dy) for slice_num, dx, dy in transformations}
    
    if verbose:
        print(f"Processing {num_slices} slices with {len(transformations)} transformations")
    
    for i in tqdm(range(num_slices), desc="Applying alignment to slices", disable=not verbose):
        slice_num = i + 1  # Assuming slices are 1-indexed
        
        if slice_num in transform_map:
            dx, dy = transform_map[slice_num]
            if verbose :
                print(f"Processing slice {slice_num}: dx={dx:.3f}, dy={dy:.3f}")
        else:
            dx, dy = 0.0, 0.0
            if verbose :
                print(f"Processing slice {slice_num}: no transformation (dx=0, dy=0)")
        
        # Apply transformation to the slice
        slice_data = image_list[i]
        aligned_slice = apply_drift_correction(slice_data, -dx, -dy)
        aligned_slices.append(aligned_slice)
    
    return aligned_slices


def save_aligned_tif(aligned_slices: List[np.ndarray], 
                    output_path: str, 
                    bit_depth: int = 8, 
                    use_lzw: bool = True, 
                    verbose: bool = False):
    """
    Save aligned slices as a single TIF file.
    
    Args:
        aligned_slices (List[np.ndarray]): List of aligned image slices
        output_path (str): Output TIF file path
        bit_depth (int): Output bit depth (8 or 16)
        use_lzw (bool): Whether to use LZW compression
        verbose (bool): Enable verbose output
    """
    if not aligned_slices:
        raise ValueError("No aligned slices to save")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if verbose:
        print(f"Saving aligned TIF file to: {output_path}")
        print(f"Number of slices: {len(aligned_slices)}")
        print(f"Slice shape: {aligned_slices[0].shape}")
    
    # Stack slices into 3D array
    aligned_array = np.stack(aligned_slices, axis=0)
    
    if verbose:
        print(f"Final array shape: {aligned_array.shape}")
    
    # Convert to appropriate data type
    if verbose:
        print("Converting data type...")
    
    if aligned_array.dtype == np.float32 or aligned_array.dtype == np.float64:
        if bit_depth == 8:
            aligned_array = np.clip(aligned_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            aligned_array = np.clip(aligned_array, 0, 65535).astype(np.uint16)
    elif aligned_array.dtype in [np.uint8, np.uint16]:
        if bit_depth == 8 and aligned_array.dtype == np.uint16:
            aligned_array = (aligned_array / 256).astype(np.uint8)
        elif bit_depth == 16 and aligned_array.dtype == np.uint8:
            aligned_array = (aligned_array.astype(np.uint16) * 256)
    else:
        if bit_depth == 8:
            aligned_array = np.clip(aligned_array, 0, 255).astype(np.uint8)
        else:  # bit_depth == 16
            aligned_array = np.clip(aligned_array, 0, 65535).astype(np.uint16)
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    # Save using tifffile with progress bar
    if verbose:
        print("Saving TIF file...")
    
    # Save using tifffile
    tifffile.imwrite(
        output_path,
        aligned_array,
        compression=compression,
        metadata={'description': 'Aligned SEM images with drift correction'}
    )
    
    if verbose:
        print(f"Successfully saved aligned TIF file: {output_path}")


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


def save_corrected_images(image_data_list: List[np.ndarray], 
                         file_paths: List[str], 
                         output_dir: str,
                         bit_depth: int = 8, 
                         use_lzw: bool = True, 
                         verbose: bool = False):
    """
    Save corrected images as individual TIF files.
    
    Args:
        image_data_list (List[np.ndarray]): List of corrected image arrays
        file_paths (List[str]): List of original file paths
        output_dir (str): Output directory path
        bit_depth (int): Output bit depth (8 or 16)
        use_lzw (bool): Whether to use LZW compression
        verbose (bool): Enable verbose output
    """
    if not image_data_list:
        print("No images to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(image_data_list)} corrected images to: {output_dir}")
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    for i, (image_data, original_path) in enumerate(zip(image_data_list, file_paths)):
        try:
            # Generate output filename
            original_name = os.path.splitext(os.path.basename(original_path))[0]
            output_filename = f"{original_name}_corrected.tif"
            output_file_path = os.path.join(output_dir, output_filename)
            
            if verbose and i % 50 == 0:
                print(f"Saving image {i+1}/{len(image_data_list)}: {output_filename}")
            
            # Convert to appropriate data type for saving
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                if bit_depth == 8:
                    image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                else:  # bit_depth == 16
                    image_data = np.clip(image_data, 0, 65535).astype(np.uint16)
            elif image_data.dtype in [np.uint8, np.uint16]:
                if bit_depth == 8 and image_data.dtype == np.uint16:
                    image_data = (image_data / 256).astype(np.uint8)
                elif bit_depth == 16 and image_data.dtype == np.uint8:
                    image_data = (image_data.astype(np.uint16) * 256)
            else:
                if bit_depth == 8:
                    image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                else:  # bit_depth == 16
                    image_data = np.clip(image_data, 0, 65535).astype(np.uint16)
            
            # Save using tifffile
            tifffile.imwrite(
                output_file_path,
                image_data,
                compression=compression,
                metadata={'description': f'Drift corrected from {os.path.basename(original_path)}'}
            )
            
        except Exception as e:
            print(f"Warning: Failed to save {output_filename}: {e}")
            continue
    
    print(f"Successfully saved {len(image_data_list)} corrected images")


def process_single_file(input_path: str, output_path: str, transformations_path: str, 
                       args, use_lzw: bool, verbose: bool, bit_depth: int):
    """Process a single TIF file."""
    try:
        print(f"Processing single file: {input_path}")
        
        # Step 1: Read the original TIF file
        print("Step 1: Reading original TIF file...")
        image_data = read_single_tif_file(input_path, verbose=verbose)
        print(f"Successfully loaded image with shape: {image_data.shape}")
        print()
        
        # Step 2: Load Results.csv file
        print("Step 2: Loading Results.csv file...")
        transformations = load_results_csv(transformations_path, verbose=verbose)
        print(f"Loaded {len(transformations)} transformations from Results.csv")
        print()
        
        # Step 3: Preprocess transformations with threshold
        print("Step 3: Preprocessing transformations...")
        preprocessed_transformations = preprocess_transformations(
            transformations, 
            args.threshold,
            args.drift_gap,
            verbose=verbose
        )
        print(f"Preprocessed {len(preprocessed_transformations)} transformations")
        print()
        
        # Step 4: Save corrected Results.csv
        corrected_csv_path = os.path.join(os.path.dirname(transformations_path), 'Results_changed.csv')
        print("Step 4: Saving corrected Results.csv...")
        save_corrected_results_csv(
            preprocessed_transformations, 
            corrected_csv_path, 
            verbose=verbose
        )
        print(f"Saved corrected transformations to: {corrected_csv_path}")
        print()
        
        # Step 5: Apply alignment to slices
        print("Step 5: Applying alignment to image slices...")
        aligned_slices = apply_alignment_to_slices(
            image_data, 
            preprocessed_transformations, 
            verbose=verbose
        )
        print(f"Successfully aligned {len(aligned_slices)} slices")
        print()
        
        # Step 6: Save final aligned TIF file
        print("Step 6: Saving final aligned TIF file...")
        save_aligned_tif(
            aligned_slices, 
            output_path, 
            bit_depth=bit_depth, 
            use_lzw=use_lzw, 
            verbose=verbose
        )
        print(f"Successfully saved aligned TIF file to: {output_path}")
        print()
        
        # Save metadata
        metadata_file = os.path.join(os.path.dirname(output_path), 'alignment_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("=== SEM Image Alignment Metadata ===\n")
            f.write(f"Original TIF file: {input_path}\n")
            f.write(f"Results CSV file: {transformations_path}\n")
            f.write(f"Corrected CSV file: {corrected_csv_path}\n")
            f.write(f"Output TIF file: {output_path}\n")
            f.write(f"Original image shape: {image_data.shape}\n")
            f.write(f"Number of slices: {len(aligned_slices)}\n")
            f.write(f"Image dtype: {image_data.dtype}\n")
            f.write(f"Output bit depth: {bit_depth} bits\n")
            f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
            f.write(f"Preprocessing threshold: {args.threshold}\n")
            f.write(f"Number of original transformations: {len(transformations)}\n")
            f.write(f"Number of preprocessed transformations: {len(preprocessed_transformations)}\n")
            f.write(f"Reference slice: {preprocessed_transformations[0][0] if preprocessed_transformations else 'Unknown'}\n")
        
        print(f"Metadata saved to: {metadata_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir: str, output_dir: str, transformations_path: str, 
                     args, use_lzw: bool, verbose: bool, bit_depth: int):
    """Process all TIF files in a directory by applying alignment directly to 2D image list (memory efficient)."""
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
        
        # Load transformations
        print("Step 2: Loading Results.csv file...")
        transformations = load_results_csv(transformations_path, verbose=verbose)
        print(f"Loaded {len(transformations)} transformations from Results.csv")
        print()
        
        # Preprocess transformations
        print("Step 3: Preprocessing transformations...")
        preprocessed_transformations = preprocess_transformations(
            transformations, 
            args.threshold,
            args.drift_gap,
            verbose=verbose
        )
        print(f"Preprocessed {len(preprocessed_transformations)} transformations")
        print()
        
        # Save corrected Results.csv
        corrected_csv_path = os.path.join(os.path.dirname(transformations_path), 'Results_changed.csv')
        print("Step 4: Saving corrected Results.csv...")
        save_corrected_results_csv(
            preprocessed_transformations, 
            corrected_csv_path, 
            verbose=verbose
        )
        print(f"Saved corrected transformations to: {corrected_csv_path}")
        print()
        
        # Apply alignment directly to 2D image list (no need to stack into 3D array)
        print("Step 5: Applying alignment to image slices...")
        aligned_slices = apply_alignment_to_slices(
            image_data_list, 
            preprocessed_transformations, 
            verbose=verbose
        )
        print(f"Successfully aligned {len(aligned_slices)} slices")
        print()
        
        # Save each slice as individual 2D TIF file
        print("Step 6: Saving aligned slices as individual 2D TIF files...")
        successful_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine compression
        compression = 'lzw' if use_lzw else None
        
        for i, slice_data in enumerate(aligned_slices):
            try:
                # Generate output filename based on original file order
                if i < len(file_paths):
                    original_name = os.path.splitext(os.path.basename(file_paths[i]))[0]
                    output_filename = f"{original_name}_aligned.tif"
                else:
                    output_filename = f"slice_{i+1:04d}_aligned.tif"
                
                output_file_path = os.path.join(output_dir, output_filename)
                
                if verbose and i % 50 == 0:
                    print(f"Saving slice {i+1}/{len(aligned_slices)}: {output_filename}")
                
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
                    metadata={'description': f'Aligned slice {i+1} from {input_dir}'}
                )
                
                successful_count += 1
                
            except Exception as e:
                print(f"Error saving slice {i+1}: {e}")
                continue
        
        print(f"Successfully saved {successful_count}/{len(aligned_slices)} aligned slices")
        print()
        
        # Save batch metadata
        metadata_file = os.path.join(output_dir, 'batch_alignment_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("=== Batch SEM Image Alignment Metadata ===\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Results CSV file: {transformations_path}\n")
            f.write(f"Corrected CSV file: {corrected_csv_path}\n")
            f.write(f"Pattern: {args.pattern}\n")
            f.write(f"Total files found: {len(file_paths)}\n")
            f.write(f"Total slices processed: {len(aligned_slices)}\n")
            f.write(f"Successfully saved slices: {successful_count}\n")
            f.write(f"Original image shape: {first_shape}\n")

            # Calculate virtual stack shape based on first_shape dimensions
            if len(first_shape) == 2:
                # 2D images: (height, width) -> virtual stack: (num_files, height, width)
                virtual_stack_shape = (len(image_data_list), first_shape[0], first_shape[1])
            elif len(first_shape) == 3:
                # 3D images: (slices, height, width) -> virtual stack: (total_slices, height, width)
                total_slices = sum(img.shape[0] if len(img.shape) == 3 else 1 for img in image_data_list)
                virtual_stack_shape = (total_slices, first_shape[1], first_shape[2])
            else:
                virtual_stack_shape = "N/A (unexpected shape dimensions)"
            f.write(f"Virtual 3D stack shape: {virtual_stack_shape}\n")

            f.write(f"Output bit depth: {bit_depth} bits\n")
            f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
            f.write(f"Preprocessing threshold: {args.threshold}\n")
            f.write(f"Number of original transformations: {len(transformations)}\n")
            f.write(f"Number of preprocessed transformations: {len(preprocessed_transformations)}\n")
        
        print(f"Batch metadata saved to: {metadata_file}")
        return True
        
    except Exception as e:
        print(f"Error processing directory {input_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to execute the alignment workflow."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression 
    
    # Handle verbose flag
    verbose = args.verbose 
    
    # Convert bit depth to integer
    bit_depth = int(args.bit_depth)
    
    # Define file paths based on command line arguments
    input_path = args.input_dir
    output_path = args.output_dir
    transformations_path = args.transformations
    
    try:
        print("=== SEM Image Alignment Workflow ===")
        print(f"Threshold for preprocessing: {args.threshold}")
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
                output_file_path = os.path.join(output_path, f"{input_name}_aligned.tif")
            else:
                output_file_path = output_path
            
            success = process_single_file(
                input_path, output_file_path, transformations_path,
                args, use_lzw, verbose, bit_depth
            )
            
        else:  # input_type == 'directory'
            # Directory processing
            if output_type == 'file':
                raise ValueError("Cannot process directory input with file output. Please specify a directory for output.")
            
            success = process_directory(
                input_path, output_path, transformations_path,
                args, use_lzw, verbose, bit_depth
            )
        
        if success:
            print("=== Alignment workflow completed successfully! ===")
        else:
            print("=== Alignment workflow completed with errors! ===")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
