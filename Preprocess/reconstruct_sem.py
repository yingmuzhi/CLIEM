#!/usr/bin/env python3
"""
Reconstruction script for processing merged TIF files.
This script reads a merged TIF file, applies vertical stretching by dividing by sin(A),
and adds pixel padding based on slice position and angle A.
"""

import argparse
import ast
import os
import math
from pathlib import Path
import tifffile
import numpy as np
from scipy.ndimage import zoom


def parse_votex_size(value):
    """Parse votex size argument formatted like '(x, y, z)'."""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            "votex_size must be provided as a tuple, e.g. '(2.458, 2.458, 8)'"
        )
    
    if (
        not isinstance(parsed, (tuple, list))
        or len(parsed) != 3
    ):
        raise argparse.ArgumentTypeError(
            "votex_size must contain exactly three numeric values"
        )
    
    try:
        return tuple(float(component) for component in parsed)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "votex_size values must be numeric"
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reconstruct merged TIF files with vertical stretching and pixel padding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters (TIF file)
  python reconstruct_sem.py --input-file /Volumes/T7/20251010_projection/src5/4_merge/drift_merged.tif --output-path ./output
  
  # Basic usage with directory containing TIF files
  python reconstruct_sem.py --input-file /path/to/tif_directory --output-path ./output
  
  # Custom angle and interpolation method
  python reconstruct_sem.py -i /path/to/input.tif -o /path/to/output --angle 45 --interpolation cubic
  
  # Different interpolation methods
  python reconstruct_sem.py -i /path/to/input.tif -o /path/to/output --interpolation nearest
  python reconstruct_sem.py -i /path/to/input.tif -o /path/to/output --interpolation linear
  python reconstruct_sem.py -i /path/to/input.tif -o /path/to/output --interpolation cubic
  
  # 16-bit output
  python reconstruct_sem.py -i /path/to/input.tif -o /path/to/output --bit-depth 16
  
  # 8-bit output (default)
  python reconstruct_sem.py -i /path/to/input.tif -o /path/to/output --bit-depth 8
        """
    )
    
    parser.add_argument(
        '--input-file', '-i',
        type=str,
        default="/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1",
        help='Input merged TIF file path or directory containing TIF files (default: /Volumes/T7/20251101_WholeCellR/Nuclei or /Volumes/T7/20251010_projection/src5/4_merge/drift_merged.tif)'
    )
    
    parser.add_argument(
        '--output-path', '-o',
        type=str,
        default='/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1_5_reconstruct',
        help='Output directory for processed files (default: ./output)'
    )
    
    parser.add_argument(
        '--angle', '-a',
        type=float,
        default=54.0,
        help='Angle A in degrees for stretching and padding calculations (default: 54.0)'
    )
    
    parser.add_argument(
        '--interpolation',
        choices=['nearest', 'linear', 'cubic'],
        default='cubic',
        help='Interpolation method for zoom operation (default: cubic)'
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
        '--bit-depth',
        choices=['8', '16'],
        default='8',
        help='Output bit depth: 8 or 16 bits (default: 8)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--votex-size',
        type=parse_votex_size,
        default=(7.9, 7.9, 8),
        help="Votex size as (x, y, z)nm; determines padding scaling (default: (2.458, 2.458, 8)nm)"
    )
    
    return parser.parse_args()


def read_merged_tif(input_path, verbose=False):
    """
    Read the merged TIF file or directory of TIF files.
    
    Args:
        input_path (str): Path to merged TIF file or directory containing TIF files
        verbose (bool): Enable verbose output
    
    Returns:
        tuple: (image_data, is_directory, file_paths)
            - image_data: numpy.ndarray (3D array for single file) or None (for directory)
            - is_directory: bool (True if input is directory, False if single file)
            - file_paths: list of file paths (only for directory, None for single file)
    """
    # Check if input path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Check if input is a directory
    if os.path.isdir(input_path):
        if verbose:
            print(f"Reading TIF files from directory: {input_path}")
        
        try:
            # Get all TIF files in the directory (ignore hidden files)
            tif_files = sorted([
                f for f in os.listdir(input_path)
                if not f.startswith('.')
                and f.lower().endswith(('.tif', '.tiff'))
            ])
            
            if not tif_files:
                raise ValueError(f"No TIF files found in directory: {input_path}")
            
            if verbose:
                print(f"Found {len(tif_files)} TIF files")
            
            # Return file paths instead of loading all images into memory
            file_paths = [os.path.join(input_path, f) for f in tif_files]
            
            if verbose:
                print(f"Will process {len(file_paths)} TIF files directly (without stacking)")
            
            return None, True, file_paths
            
        except Exception as e:
            raise Exception(f"Failed to read TIF files from directory {input_path}: {e}")
    
    else:
        # It's a file (could be single 3D TIF or multi-page TIF)
        if verbose:
            print(f"Reading merged TIF file: {input_path}")
        
        try:
            # Read TIF file using tifffile
            image_data = tifffile.imread(input_path)
            
            if verbose:
                print(f"Successfully loaded merged TIF file")
                print(f"Image shape: {image_data.shape}")
                print(f"Image dtype: {image_data.dtype}")
                if image_data.ndim == 3:
                    print(f"Number of slices: {image_data.shape[0]}")
            
            return image_data, False, None
            
        except Exception as e:
            raise Exception(f"Failed to read {input_path}: {e}")


def compute_z_ratio(votex_size):
    """Compute z_ratio from votex size."""
    vx, vy, vz = votex_size
    if not math.isclose(vx, vy, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("votex_size x and y components must be equal")
    if vx == 0:
        raise ValueError("votex_size x (and y) component must be non-zero")
    return vz / vx


def apply_vertical_stretching_and_padding(image_data=None, file_paths=None, num_slices=None, 
                                         angle_degrees=54.0, interpolation='cubic', z_ratio=1.0, verbose=False):
    """
    Lazily yield per-slice arrays after vertical stretching (factor 1/sin(A)) and padding.
    Processing happens slice-by-slice so that we never keep the entire volume in memory.
    
    Args:
        image_data (numpy.ndarray, optional): Input image data (3D array: slices, height, width)
            Used when input is a single 3D TIF file
        file_paths (list, optional): List of file paths to 2D TIF files
            Used when input is a directory of 2D files
        num_slices (int, optional): Number of slices (required if file_paths is provided)
        angle_degrees (float): Angle A in degrees
        interpolation (str): Interpolation method for zoom
        z_ratio (float): z-axis ratio for padding calculation
        verbose (bool): Enable verbose output
    
    Yields:
        numpy.ndarray: Processed 2D image array with padding
    """
    # Geometry calculations
    angle_rad = math.radians(angle_degrees)
    sine_a = math.sin(angle_rad)
    tan_a = math.tan(angle_rad)
    
    if sine_a == 0:
        raise ValueError("Angle results in zero sine; cannot stretch vertically.")
    
    stretch_factor = 1.0 / sine_a
    
    if verbose:
        print(f"Angle: {angle_degrees} degrees")
        print(f"sin(A) = {sine_a:.6f}")
        print(f"tan(A) = {tan_a:.6f}")
        print(f"Vertical stretch factor = {stretch_factor:.6f}")
        print(f"Interpolation method: {interpolation}")
    
    # Map interpolation string to scipy order
    interpolation_map = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3
    }
    zoom_order = interpolation_map[interpolation]
    
    # Determine number of slices and processing mode
    if file_paths is not None:
        # Process from file paths (directory mode)
        if num_slices is None:
            num_slices = len(file_paths)
        process_from_files = True
    elif image_data is not None:
        # Process from 3D array (single file mode)
        if image_data.ndim != 3:
            raise ValueError(f"Expected 3D image data (slices, height, width), got {image_data.ndim}D")
        num_slices = image_data.shape[0]
        original_width = image_data.shape[2]
        process_from_files = False
    else:
        raise ValueError("Either image_data or file_paths must be provided")
    
    # Process each slice lazily
    for n in range(num_slices):
        if verbose and n % 50 == 0:
            print(f"Processing slice {n+1}/{num_slices}")
        
        # Load current slice
        if process_from_files:
            # Read 2D file directly
            current_slice = tifffile.imread(file_paths[n])
            if current_slice.ndim == 3:
                # Multi-page TIF, take first page
                current_slice = current_slice[0]
            elif current_slice.ndim != 2:
                raise ValueError(f"Unexpected image dimensions in {file_paths[n]}: {current_slice.ndim}D")
            original_width = current_slice.shape[1]
        else:
            # Get slice from 3D array
            current_slice = image_data[n, :, :]
        
        # Apply vertical stretching using zoom
        stretched_slice = zoom(current_slice, (stretch_factor, 1), order=zoom_order)
        stretched_height = stretched_slice.shape[0]
        
        # Calculate padding
        # Bottom padding: round(n/tan(A))
        bottom_padding = int(round(z_ratio * n / tan_a))    # add for pixel size(0.004nm, 0.004nm, 0.008nm)
        
        # Top padding: round(num/tan(A)) - round(n/tan(A))
        top_padding = int(round(z_ratio * num_slices / tan_a)) - int(round(z_ratio * n / tan_a))    # add for pixel size(0.004nm, 0.004nm, 0.008nm)
        
        # Ensure padding is non-negative
        bottom_padding = max(0, bottom_padding)
        top_padding = max(0, top_padding)
        
        if verbose and n < 5:  # Show details for first 5 slices
            print(f"  Slice {n}: bottom_padding={bottom_padding}, top_padding={top_padding}")
        
        # Create padded image
        total_height = top_padding + stretched_height + bottom_padding
        padded_image = np.zeros((total_height, original_width), dtype=current_slice.dtype)
        
        # Place stretched slice in the middle
        start_row = top_padding
        end_row = start_row + stretched_height
        padded_image[start_row:end_row, :] = stretched_slice
        
        yield padded_image
    
    if verbose:
        print("Finished processing all slices")


def save_processed_images(image_iterable, output_path, input_filename, use_lzw=True,
                          bit_depth=8, total_images=None, verbose=False):
    """
    Save processed images as individual TIF files while consuming them lazily.
    
    Args:
        image_iterable (Iterable[np.ndarray]): Iterable/generator of processed 2D image arrays
        output_path (str): Output directory path
        input_filename (str): Original input filename (for naming)
        use_lzw (bool): Whether to use LZW compression
        bit_depth (int): Output bit depth (8 or 16)
        total_images (int, optional): Expected number of output slices for progress display
        verbose (bool): Enable verbose output
    
    Returns:
        tuple: (saved_count, sample_shape)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving processed images to: {output_path}")
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    # Get base name from input file
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    saved_count = 0
    sample_shape = None
    
    for i, image_data in enumerate(image_iterable):
        try:
            # Generate output filename
            output_filename = f"{base_name}_slice_{i:04d}_processed.tif"
            output_file_path = os.path.join(output_path, output_filename)
            
            saved_count_preview = i + 1
            if total_images:
                progress_pct = (saved_count_preview / total_images) * 100
                if verbose or saved_count_preview == 1 or saved_count_preview % 50 == 0 or saved_count_preview == total_images:
                    print(f"Saving image {saved_count_preview}/{total_images} ({progress_pct:.1f}%) -> {output_filename}")
            else:
                if verbose or saved_count_preview == 1 or saved_count_preview % 50 == 0:
                    print(f"Saving image {saved_count_preview} -> {output_filename}")
            
            sample_shape = sample_shape or image_data.shape
            
            # Convert to appropriate data type for saving (avoid mutating original array)
            if image_data.dtype in (np.float32, np.float64):
                if bit_depth == 8:
                    image_to_save = np.clip(image_data, 0, 255).astype(np.uint8)
                else:  # bit_depth == 16
                    image_to_save = np.clip(image_data, 0, 65535).astype(np.uint16)
            elif image_data.dtype == np.uint8 and bit_depth == 16:
                image_to_save = (image_data.astype(np.uint16) * 256)
            elif image_data.dtype == np.uint16 and bit_depth == 8:
                image_to_save = (image_data / 256).astype(np.uint8)
            else:
                target_dtype = np.uint8 if bit_depth == 8 else np.uint16
                clip_max = 255 if bit_depth == 8 else 65535
                image_to_save = np.clip(image_data, 0, clip_max).astype(target_dtype)
            
            # Save using tifffile
            tifffile.imwrite(
                output_file_path,
                image_to_save,
                compression=compression,
                metadata={'description': f'Processed slice {i} from {os.path.basename(input_filename)}'}
            )
            saved_count += 1
            
        except Exception as e:
            print(f"Warning: Failed to save {output_filename}: {e}")
            continue
    
    print(f"Successfully saved {saved_count} processed images")
    return saved_count, sample_shape


def main():
    """Main function to execute the reconstruction process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression and not args.no_lzw_compression
    
    # Convert bit depth to integer
    bit_depth = int(args.bit_depth)
    
    try:
        # Read merged TIF file or directory of TIF files
        if os.path.isdir(args.input_file):
            print(f"Reading TIF files from directory: {args.input_file}")
        else:
            print(f"Reading merged TIF file from: {args.input_file}")
        image_data, is_directory, file_paths = read_merged_tif(
            input_path=args.input_file,
            verbose=args.verbose
        )
        
        # Get number of slices and original shape info
        if is_directory:
            num_slices = len(file_paths)
            # Read first file to get shape info
            first_slice = tifffile.imread(file_paths[0])
            if first_slice.ndim == 3:
                first_slice = first_slice[0]
            original_shape = (num_slices, first_slice.shape[0], first_slice.shape[1])
            original_dtype = first_slice.dtype
            print(f"Found {num_slices} TIF files in directory")
            print(f"Each file shape: {first_slice.shape}")
        else:
            num_slices = image_data.shape[0]
            original_shape = image_data.shape
            original_dtype = image_data.dtype
            print(f"Loaded merged TIF with {num_slices} slices")
            print(f"Original image shape: {original_shape}")
        
        print(f"Output directory: {args.output_path}")
        print(f"Angle A: {args.angle} degrees")
        print(f"Interpolation method: {args.interpolation}")
        print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
        print(f"Output bit depth: {bit_depth} bits")
        
        z_ratio = compute_z_ratio(args.votex_size)
        
        if args.verbose:
            print(f"Votex size: {args.votex_size}")
            print(f"z_ratio (vz/vx): {z_ratio}")
        
        # Apply vertical stretching and padding
        print("Applying vertical stretching and pixel padding...")
        if is_directory:
            # Process 2D files directly without stacking
            processed_image_iter = apply_vertical_stretching_and_padding(
                file_paths=file_paths,
                num_slices=num_slices,
                angle_degrees=args.angle,
                interpolation=args.interpolation,
                z_ratio=z_ratio,
                verbose=args.verbose
            )
        else:
            # Process from 3D array
            processed_image_iter = apply_vertical_stretching_and_padding(
                image_data=image_data,
                angle_degrees=args.angle,
                interpolation=args.interpolation,
                z_ratio=z_ratio,
                verbose=args.verbose
            )
        
        # Save processed images
        print("Saving processed images...")
        saved_count, sample_shape = save_processed_images(
            processed_image_iter,
            args.output_path,
            args.input_file,
            use_lzw=use_lzw,
            bit_depth=bit_depth,
            total_images=num_slices,
            verbose=args.verbose
        )
        
        # Save metadata
        metadata_file = os.path.join(args.output_path, 'metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Input file: {args.input_file}\n")
            f.write(f"Input type: {'Directory' if is_directory else 'Single 3D TIF file'}\n")
            f.write(f"Number of slices: {num_slices}\n")
            f.write(f"Original image shape: {original_shape}\n")
            if sample_shape is not None:
                f.write(f"Processed image shape: {sample_shape}\n")
            else:
                f.write("Processed image shape: N/A (no slices were saved)\n")
            f.write(f"Image dtype: {original_dtype}\n")
            f.write(f"Angle A: {args.angle} degrees\n")
            f.write(f"sin(A): {math.sin(math.radians(args.angle)):.6f}\n")
            f.write(f"tan(A): {math.tan(math.radians(args.angle)):.6f}\n")
            f.write(f"Vertical stretch factor: {1.0/math.sin(math.radians(args.angle)):.6f}\n")
            f.write(f"Interpolation method: {args.interpolation}\n")
            f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
            f.write(f"Output bit depth: {bit_depth} bits\n")
            f.write(f"Votex size: {args.votex_size}\n")
            f.write(f"z_ratio (vz/vx): {z_ratio}\n")
            f.write(f"Processing details:\n")
            f.write(f"  - Each slice is vertically stretched by factor 1/sin(A)\n")
            f.write(f"  - Bottom padding for slice n: round(z_ratio * n/tan(A))\n")
            f.write(f"  - Top padding for slice n: round(z_ratio * num/tan(A)) - round(z_ratio * n/tan(A))\n")
            if is_directory:
                f.write(f"  - Processed 2D files directly without stacking into 3D array\n")
        
        print(f"Metadata saved to: {metadata_file}")
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
