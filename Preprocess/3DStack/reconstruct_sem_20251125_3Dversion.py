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
        default="/Volumes/T7/20251010_halfCell/preprocessedData_Part/4_merge/drift_merged.tif",
        help='Input merged TIF file path or directory containing TIF files (default: /Volumes/T7/20251101_WholeCellR/Nuclei or /Volumes/T7/20251010_projection/src5/4_merge/drift_merged.tif)'
    )
    
    parser.add_argument(
        '--output-path', '-o',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/5_reconstruct',
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
        default=(2.458, 2.458, 8),
        help="Votex size as (x, y, z)nm; determines padding scaling (default: (2.458, 2.458, 8)nm)"
    )
    
    return parser.parse_args()


def read_merged_tif(input_path, verbose=False):
    """
    Read the merged TIF file or directory of TIF files and return image data.
    
    Args:
        input_path (str): Path to merged TIF file or directory containing TIF files
        verbose (bool): Enable verbose output
    
    Returns:
        numpy.ndarray: Loaded image data (3D array: slices, height, width)
    """
    # Check if input path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Check if input is a directory
    if os.path.isdir(input_path):
        if verbose:
            print(f"Reading TIF files from directory: {input_path}")
        
        try:
            # Get all TIF files in the directory
            tif_files = sorted([f for f in os.listdir(input_path) 
                              if f.lower().endswith(('.tif', '.tiff'))])
            
            if not tif_files:
                raise ValueError(f"No TIF files found in directory: {input_path}")
            
            if verbose:
                print(f"Found {len(tif_files)} TIF files")
            
            # Read all TIF files and stack them
            image_slices = []
            for tif_file in tif_files:
                file_path = os.path.join(input_path, tif_file)
                if verbose and len(image_slices) % 50 == 0:
                    print(f"Reading file {len(image_slices)+1}/{len(tif_files)}: {tif_file}")
                
                slice_data = tifffile.imread(file_path)
                
                # Handle 2D images
                if slice_data.ndim == 2:
                    image_slices.append(slice_data)
                elif slice_data.ndim == 3:
                    # If it's a 3D array (multi-page TIF), take first page or stack them
                    # For simplicity, we'll take all pages
                    for page in range(slice_data.shape[0]):
                        image_slices.append(slice_data[page])
                else:
                    raise ValueError(f"Unexpected image dimensions in {tif_file}: {slice_data.ndim}D")
            
            # Stack all slices into 3D array
            image_data = np.stack(image_slices, axis=0)
            
            if verbose:
                print(f"Successfully loaded {len(tif_files)} TIF files from directory")
                print(f"Image shape: {image_data.shape}")
                print(f"Image dtype: {image_data.dtype}")
                print(f"Number of slices: {image_data.shape[0]}")
            
            return image_data
            
        except Exception as e:
            raise Exception(f"Failed to read TIF files from directory {input_path}: {e}")
    
    else:
        # It's a file
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
            
            return image_data
            
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


def apply_vertical_stretching_and_padding(image_data, angle_degrees, interpolation='cubic', z_ratio=1.0, verbose=False):
    """
    Apply vertical stretching by factor 1/sin(A) and add per-slice padding.
    
    Args:
        image_data (numpy.ndarray): Input image data (3D array: slices, height, width)
        angle_degrees (float): Angle A in degrees
        interpolation (str): Interpolation method for zoom
        verbose (bool): Enable verbose output
    
    Returns:
        list: List of processed 2D image arrays with padding
    """
    if image_data.ndim != 3:
        raise ValueError(f"Expected 3D image data (slices, height, width), got {image_data.ndim}D")
    
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
    
    num_slices, original_height, original_width = image_data.shape
    
    # Map interpolation string to scipy order
    interpolation_map = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3
    }
    zoom_order = interpolation_map[interpolation]
    
    # Process each slice
    processed_images = []
    
    for n in range(num_slices):
        if verbose and n % 50 == 0:
            print(f"Processing slice {n+1}/{num_slices}")
        
        # Get current slice
        current_slice = image_data[n, :, :]
        
        # Apply vertical stretching using zoom
        stretched_slice = zoom(current_slice, (stretch_factor, 1), order=zoom_order)
        stretched_height = stretched_slice.shape[0]
        
        # Calculate padding
        # Bottom padding: round(n/tan(A))
        bottom_padding = (  int(round(z_ratio * n / tan_a))  )    # add for pixel size(0.004nm, 0.004nm, 0.008nm)
        
        # Top padding: round(num/tan(A)) - round(n/tan(A))
        top_padding =  (  int(round(z_ratio * num_slices / tan_a)) - int(round(z_ratio * n / tan_a))  )    # add for pixel size(0.004nm, 0.004nm, 0.008nm)
        
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
        
        processed_images.append(padded_image)
    
    if verbose:
        print(f"Processed {len(processed_images)} slices")
        if processed_images:
            print(f"Output image shape: {processed_images[0].shape}")
    
    return processed_images


def save_processed_images(image_data_list, output_path, input_filename, use_lzw=True, bit_depth=8, verbose=False):
    """
    Save processed images as individual TIF files.
    
    Args:
        image_data_list (list): List of processed 2D image arrays
        output_path (str): Output directory path
        input_filename (str): Original input filename (for naming)
        use_lzw (bool): Whether to use LZW compression
        bit_depth (int): Output bit depth (8 or 16)
        verbose (bool): Enable verbose output
    """
    if not image_data_list:
        print("No images to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving {len(image_data_list)} processed images to: {output_path}")
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    # Get base name from input file
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    for i, image_data in enumerate(image_data_list):
        try:
            # Generate output filename
            output_filename = f"{base_name}_slice_{i:04d}_processed.tif"
            output_file_path = os.path.join(output_path, output_filename)
            
            if verbose and i % 50 == 0:
                print(f"Saving image {i+1}/{len(image_data_list)}: {output_filename}")
            
            # Convert to appropriate data type for saving
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                if bit_depth == 8:
                    # Convert float to uint8 (0-255 range)
                    image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                else:  # bit_depth == 16
                    # Convert float to uint16 (0-65535 range)
                    image_data = np.clip(image_data, 0, 65535).astype(np.uint16)
            elif image_data.dtype in [np.uint8, np.uint16]:
                # If already in correct bit depth, keep as is
                if bit_depth == 8 and image_data.dtype == np.uint16:
                    # Convert from 16-bit to 8-bit
                    image_data = (image_data / 256).astype(np.uint8)
                elif bit_depth == 16 and image_data.dtype == np.uint8:
                    # Convert from 8-bit to 16-bit
                    image_data = (image_data.astype(np.uint16) * 256)
            else:
                # For other data types, convert based on bit depth
                if bit_depth == 8:
                    image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                else:  # bit_depth == 16
                    image_data = np.clip(image_data, 0, 65535).astype(np.uint16)
            
            # Save using tifffile
            tifffile.imwrite(
                output_file_path,
                image_data,
                compression=compression,
                metadata={'description': f'Processed slice {i} from {os.path.basename(input_filename)}'}
            )
            
        except Exception as e:
            print(f"Warning: Failed to save {output_filename}: {e}")
            continue
    
    print(f"Successfully saved {len(image_data_list)} processed images")


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
        image_data = read_merged_tif(
            input_path=args.input_file,
            verbose=args.verbose
        )
        
        print(f"Loaded merged TIF with {image_data.shape[0]} slices")
        print(f"Original image shape: {image_data.shape}")
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
        processed_images = apply_vertical_stretching_and_padding(
            image_data, 
            args.angle,
            args.interpolation,
            z_ratio=z_ratio,
            verbose=args.verbose
        )
        
        # Save processed images
        print("Saving processed images...")
        save_processed_images(
            processed_images,
            args.output_path,
            args.input_file,
            use_lzw=use_lzw,
            bit_depth=bit_depth,
            verbose=args.verbose
        )
        
        # Save metadata
        metadata_file = os.path.join(args.output_path, 'metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Input file: {args.input_file}\n")
            f.write(f"Number of slices: {image_data.shape[0]}\n")
            f.write(f"Original image shape: {image_data.shape}\n")
            f.write(f"Processed image shape: {processed_images[0].shape}\n")
            f.write(f"Image dtype: {image_data.dtype}\n")
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
            f.write(f"  - Bottom padding for slice n: round(n/tan(A))\n")
            f.write(f"  - Top padding for slice n: round(num/tan(A)) - round(n/tan(A))\n")
        
        print(f"Metadata saved to: {metadata_file}")
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
