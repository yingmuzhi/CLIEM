'''
Author: yingmuzhi cyxscj@126.com
Date: 2025-10-20 09:41:19
LastEditors: yingmuzhi cyxscj@126.com
LastEditTime: 2025-11-18 18:02:43
FilePath: /20251010_projection/code/flip_sem.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python3
"""
Flip SEM images script.
This script reads TIF files from part1_output directory, applies horizontal and vertical flipping,
and saves the results to part1_output_flip directory.
"""

import argparse
import os
import glob
from pathlib import Path
import tifffile
import natsort
import numpy as np


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Flip SEM images horizontally and vertically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python flip_sem.py
  
  # Custom input and output directories
  python flip_sem.py -i /path/to/input -o /path/to/output
  
  # Process single file (can be multi-layer stack)
  python flip_sem.py -i /path/to/single_file.tif -o /path/to/output
  
  # Verbose output
  python flip_sem.py --verbose
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/5_reconstruct',
        help='Input directory containing TIF files or single TIF file (can be multi-layer stack) (default: /Volumes/T7/20251010_projection/src4/part1_output)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/Volumes/T7/20251010_halfCell/preprocessedData_Part/6_flip',
        help='Output directory for flipped files (default: /Volumes/T7/20251010_projection/src4/part1_output_flip)'
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
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def read_tif_files(input_path, pattern='*.tif', verbose=False):
    """
    Read TIF files from the specified path (directory or single file) and handle multi-layer stacks.
    
    Args:
        input_path (str): Path to directory containing TIF files or single TIF file
        pattern (str): File pattern to match (default: '*.tif') - only used for directories
        verbose (bool): Enable verbose output
    
    Returns:
        tuple: (file_paths, image_data_list, is_stack) where:
               - file_paths is list of file paths (for single file, contains one path)
               - image_data_list is list of loaded image arrays (for stacks, each slice is separate)
               - is_stack is boolean indicating if input was a multi-layer stack
    """
    # Check if input path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Check if input is a single file or directory
    if os.path.isfile(input_path):
        # Single file input
        if verbose:
            print(f"Processing single file: {input_path}")
        
        try:
            # Read the TIF file
            image_data = tifffile.imread(input_path)
            
            # Check if it's a multi-layer stack
            if len(image_data.shape) == 3:
                # Multi-layer stack - separate each slice
                if verbose:
                    print(f"Detected multi-layer stack with {image_data.shape[0]} slices")
                    print(f"Stack shape: {image_data.shape}")
                
                # Create separate arrays for each slice
                image_data_list = []
                file_paths = []
                
                for i in range(image_data.shape[0]):
                    slice_data = image_data[i]
                    image_data_list.append(slice_data)
                    # Create a virtual file path for each slice
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    slice_filename = f"{base_name}_slice_{i:04d}.tif"
                    file_paths.append(slice_filename)
                
                if verbose:
                    print(f"Separated {len(image_data_list)} slices from stack")
                
                return file_paths, image_data_list, True
                
            else:
                # Single image
                if verbose:
                    print(f"Detected single image with shape: {image_data.shape}")
                
                return [input_path], [image_data], False
                
        except Exception as e:
            raise Exception(f"Failed to read file {input_path}: {e}")
    
    else:
        # Directory input - original behavior
        if verbose:
            print(f"Processing directory: {input_path}")
        
        # Find all TIF files matching the pattern
        search_pattern = os.path.join(input_path, pattern)
        tif_files = glob.glob(search_pattern)
        
        if not tif_files:
            raise FileNotFoundError(f"No TIF files found in {input_path} matching pattern '{pattern}'")
        
        if verbose:
            print(f"Found {len(tif_files)} TIF files in {input_path}")
        
        # Sort files naturally using natsort
        sorted_files = natsort.natsorted(tif_files)
        
        if verbose:
            print("Files sorted naturally:")
            for i, file_path in enumerate(sorted_files[:5]):  # Show first 5 files
                print(f"  {i+1}: {os.path.basename(file_path)}")
            if len(sorted_files) > 5:
                print(f"  ... and {len(sorted_files) - 5} more files")
        
        # Read all TIF files using tifffile
        image_data_list = []
        for i, file_path in enumerate(sorted_files):
            try:
                if verbose and i % 50 == 0:  # Progress indicator every 50 files
                    print(f"Reading file {i+1}/{len(sorted_files)}: {os.path.basename(file_path)}")
                
                # Read TIF file using tifffile
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
        
        return sorted_files, image_data_list, False


def flip_images(image_data_list, verbose=False):
    """
    Apply horizontal and vertical flipping to all images.
    
    Args:
        image_data_list (list): List of image arrays
        verbose (bool): Enable verbose output
    
    Returns:
        list: List of flipped image arrays
    """
    if not image_data_list:
        return []
    
    flipped_images = []
    
    for i, image in enumerate(image_data_list):
        if verbose and i % 50 == 0:
            print(f"Flipping image {i+1}/{len(image_data_list)}")
        
        # Apply both horizontal and vertical flipping
        # np.flip with axis=(0,1) flips both horizontally and vertically
        flipped_image = np.flip(image, axis=(0, 1))
        flipped_images.append(flipped_image)
    
    return flipped_images


def save_flipped_images(image_data_list, file_paths, output_dir, bit_depth=8, use_lzw=True, verbose=False, is_stack=False):
    """
    Save flipped images as individual TIF files.
    
    Args:
        image_data_list (list): List of flipped image arrays
        file_paths (list): List of original file paths
        output_dir (str): Output directory path
        bit_depth (int): Output bit depth (8 or 16)
        use_lzw (bool): Whether to use LZW compression
        verbose (bool): Enable verbose output
        is_stack (bool): Whether the input was a multi-layer stack
    """
    if not image_data_list:
        print("No images to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(image_data_list)} flipped images to: {output_dir}")
    
    # Determine compression
    compression = 'lzw' if use_lzw else None
    
    for i, (image_data, original_path) in enumerate(zip(image_data_list, file_paths)):
        try:
            # Generate output filename
            if is_stack:
                # For stack slices, use the virtual filename directly
                original_name = os.path.splitext(original_path)[0]
                output_filename = f"{original_name}_flipped.tif"
            else:
                # For regular files, add _flipped suffix
                original_name = os.path.splitext(os.path.basename(original_path))[0]
                output_filename = f"{original_name}_flipped.tif"
            
            output_file_path = os.path.join(output_dir, output_filename)
            
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
                metadata={'description': f'Flipped from {os.path.basename(original_path)}'}
            )
            
        except Exception as e:
            print(f"Warning: Failed to save {output_filename}: {e}")
            continue
    
    print(f"Successfully saved {len(image_data_list)} flipped images")


def main():
    """Main function to execute the flipping process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression and not args.no_lzw_compression
    
    # Convert bit depth to integer
    bit_depth = int(args.bit_depth)
    
    try:
        # Read and sort TIF files
        print(f"Reading TIF files from: {args.input_dir}")
        file_paths, image_data_list, is_stack = read_tif_files(
            input_path=args.input_dir,
            pattern=args.pattern,
            verbose=args.verbose
        )
        
        if not image_data_list:
            print("No images loaded successfully")
            return 1
        
        print(f"Loaded {len(image_data_list)} images")
        if is_stack:
            print("Processing multi-layer stack - each slice will be saved separately")
        print(f"Output directory: {args.output_dir}")
        print(f"Output bit depth: {bit_depth} bits")
        print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
        
        # Apply flipping transformations
        print("Applying horizontal and vertical flipping...")
        flipped_images = flip_images(image_data_list, verbose=args.verbose)
        
        # Save flipped images
        print("Saving flipped images...")
        save_flipped_images(
            flipped_images,
            file_paths,
            args.output_dir,
            bit_depth=bit_depth,
            use_lzw=use_lzw,
            verbose=args.verbose,
            is_stack=is_stack
        )
        
        # Save metadata
        metadata_file = os.path.join(args.output_dir, 'flip_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Number of images: {len(image_data_list)}\n")
            f.write(f"Input type: {'Multi-layer stack' if is_stack else 'Directory of files'}\n")
            f.write(f"Original image shape: {image_data_list[0].shape}\n")
            f.write(f"Flipped image shape: {flipped_images[0].shape}\n")
            f.write(f"Image dtype: {image_data_list[0].dtype}\n")
            f.write(f"Output bit depth: {bit_depth} bits\n")
            f.write(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}\n")
            f.write(f"Transformation: Horizontal and vertical flip\n")
            f.write(f"File list:\n")
            for i, file_path in enumerate(file_paths):
                f.write(f"{i+1}: {os.path.basename(file_path)}\n")
        
        print(f"Metadata saved to: {metadata_file}")
        print("Flipping completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
