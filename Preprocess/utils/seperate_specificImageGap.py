#!/usr/bin/env python3
"""
Script to separate specific image gaps between parts.
Reads TIF files from input directory, extracts the last image from each part
and the first image from the next part, then saves them as gap images.
"""

import argparse
import os
import glob
import numpy as np
import tifffile
import json
from pathlib import Path
import natsort
from typing import List, Tuple, Union


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Separate specific image gaps between parts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default directories
  python seperate_specificImageGap.py
  
  # Custom input and output directories
  python seperate_specificImageGap.py -i /path/to/input -o /path/to/output
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='/Volumes/T7/20251010_projection/src5/2_resize/drift',
        help='Input directory containing TIF files with part names'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='/Volumes/T7/20251010_projection/src5/2_resize/gap_align',
        help='Output directory for gap images'
    )
    
    parser.add_argument(
        '--lzw-compression', action='store_true', default=True,
        help='Use LZW compression for output files (default: True)'
    )
    
    parser.add_argument(
        '--no-lzw-compression', action='store_true',
        help='Disable LZW compression for output files'
    )
    
    return parser.parse_args()


def get_part_files(input_dir: str) -> dict:
    """
    Get all TIF files grouped by part number.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        dict: Dictionary with part numbers as keys and file lists as values
    """
    # Get all TIF files in the directory
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    # Group files by part number
    part_files = {}
    
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        
        # Look for 'part' followed by a number in the filename
        if 'part' in filename.lower():
            # Extract part number using simple string manipulation
            parts = filename.lower().split('part')
            if len(parts) > 1:
                # Get the part number (first digit sequence after 'part')
                part_str = parts[1]
                part_num = ''
                for char in part_str:
                    if char.isdigit():
                        part_num += char
                    else:
                        break
                
                if part_num:
                    part_number = int(part_num)
                    if part_number not in part_files:
                        part_files[part_number] = []
                    part_files[part_number].append(file_path)
    
    # Sort files within each part
    for part_num in part_files:
        part_files[part_num] = natsort.natsorted(part_files[part_num])
    
    return part_files


def is_tif_stack(image_path: str) -> bool:
    """Check if a TIF file is a stack (multi-layer)."""
    try:
        with tifffile.TiffFile(image_path) as tif:
            return len(tif.pages) > 1
    except Exception as e:
        print(f"Error checking stack status for {image_path}: {e}")
        return False


def get_image_from_file(file_path: str, frame_index: int = 0):
    """
    Get a specific frame from a TIF file.
    
    Args:
        file_path: Path to the TIF file
        frame_index: Index of the frame to extract (0 for first, -1 for last)
        
    Returns:
        numpy array: Image data
    """
    try:
        with tifffile.TiffFile(file_path) as tif:
            if len(tif.pages) == 1:
                # Single frame
                return tif.pages[0].asarray()
            else:
                # Multi-frame stack
                if frame_index == -1:
                    # Get last frame
                    return tif.pages[-1].asarray()
                else:
                    # Get specific frame
                    return tif.pages[frame_index].asarray()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def save_gap_images(image1: np.ndarray, image2: np.ndarray, output_dir: str, gap_name: str, part1: int, part2: int, use_lzw: bool = True):
    """
    Save two images as separate files in the gap directory.
    
    Args:
        image1: First image (last image from previous part)
        image2: Second image (first image from next part)
        output_dir: Output directory for gap images
        gap_name: Name of the gap (e.g., "gap1_part1ToPart2")
        part1: First part number
        part2: Second part number
        use_lzw: Whether to use LZW compression
    """
    try:
        print(f"  Image1 shape: {image1.shape}, dtype: {image1.dtype}")
        print(f"  Image2 shape: {image2.shape}, dtype: {image2.dtype}")
        
        # Determine compression
        compression = 'lzw' if use_lzw else None
        
        # Save first image (last image from previous part)
        output_path1 = os.path.join(output_dir, f"{gap_name}_part{part1}.tif")
        tifffile.imwrite(
            output_path1,
            image1,
            compression=compression,
            metadata={'description': f'Last image from part{part1}'}
        )
        print(f"Saved last image from part{part1} to: {output_path1}")
        
        # Save second image (first image from next part)
        output_path2 = os.path.join(output_dir, f"{gap_name}_part{part2}.tif")
        tifffile.imwrite(
            output_path2,
            image2,
            compression=compression,
            metadata={'description': f'First image from part{part2}'}
        )
        print(f"Saved first image from part{part2} to: {output_path2}")
        
        # Verify the saved files
        with tifffile.TiffFile(output_path1) as tif:
            print(f"  File1 has {len(tif.pages)} pages, shape: {tif.pages[0].asarray().shape}")
        
        with tifffile.TiffFile(output_path2) as tif:
            print(f"  File2 has {len(tif.pages)} pages, shape: {tif.pages[0].asarray().shape}")
        
    except Exception as e:
        print(f"Error saving gap images to {output_dir}: {e}")


def process_gaps(part_files: dict, output_dir: str, use_lzw: bool = True):
    """
    Process gaps between parts.
    
    Args:
        part_files: Dictionary of part files
        output_dir: Output directory
        use_lzw: Whether to use LZW compression
    """
    # Get sorted part numbers
    part_numbers = sorted(part_files.keys())
    
    if len(part_numbers) < 2:
        print("Need at least 2 parts to create gaps")
        return
    
    print(f"Found {len(part_numbers)} parts: {part_numbers}")
    
    # Process gaps between consecutive parts
    for i in range(len(part_numbers) - 1):
        current_part = part_numbers[i]
        next_part = part_numbers[i + 1]
        
        print(f"\nProcessing gap between part{current_part} and part{next_part}")
        
        # Get files for current and next parts
        current_files = part_files[current_part]
        next_files = part_files[next_part]
        
        if not current_files or not next_files:
            print(f"Skipping gap {current_part}->{next_part}: missing files")
            continue
        
        print(f"  Part{current_part}: {len(current_files)} files")
        print(f"  Part{next_part}: {len(next_files)} files")
        
        # Get last image from current part
        last_file = current_files[-1]
        print(f"  Last file from part{current_part}: {os.path.basename(last_file)}")
        
        # Get first image from next part
        first_file = next_files[0]
        print(f"  First file from part{next_part}: {os.path.basename(first_file)}")
        
        # Extract images
        last_image = get_image_from_file(last_file, -1)  # Last frame
        first_image = get_image_from_file(first_file, 0)  # First frame
        
        if last_image is None or first_image is None:
            print(f"Skipping gap {current_part}->{next_part}: failed to read images")
            continue
        
        print(f"  Successfully loaded images:")
        print(f"    Last image from part{current_part}: shape {last_image.shape}")
        print(f"    First image from part{next_part}: shape {first_image.shape}")
        
        # Create output directory for this gap
        gap_dir = os.path.join(output_dir, f"gap{i+1}_part{current_part}ToPart{next_part}")
        os.makedirs(gap_dir, exist_ok=True)
        
        # Save gap images as separate files
        gap_name = f"gap{i+1}_part{current_part}ToPart{next_part}"
        save_gap_images(last_image, first_image, gap_dir, gap_name, current_part, next_part, use_lzw)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression and not args.no_lzw_compression
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
    print("-" * 50)
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get part files
    print("Scanning for part files...")
    part_files = get_part_files(args.input_dir)
    
    if not part_files:
        print("No part files found in input directory")
        return
    
    print(f"Found files for parts: {sorted(part_files.keys())}")
    
    # Process gaps
    process_gaps(part_files, args.output_dir, use_lzw)
    
    print("\nGap processing completed!")
    
    # Save processing information
    info_path = os.path.join(args.output_dir, 'gap_processing_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            'input_directory': args.input_dir,
            'output_directory': args.output_dir,
            'lzw_compression': use_lzw,
            'parts_found': sorted(part_files.keys()),
            'part_file_counts': {str(k): len(v) for k, v in part_files.items()},
            'gaps_processed': len(part_files) - 1 if len(part_files) > 1 else 0
        }, f, indent=2)
    
    print(f"Processing information saved to: {info_path}")


if __name__ == "__main__":
    main()
