#!/usr/bin/env python3
"""
Script to calculate min/max width and height of multiple TIFF files.
"""

import os
import sys
from PIL import Image
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Dict


def read_tiff_dimensions(file_path: str) -> Tuple[int, int]:
    """
    Read TIFF file and return its width and height.
    
    Args:
        file_path (str): Path to the TIFF file
        
    Returns:
        Tuple[int, int]: (width, height) of the image
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If the file cannot be read as an image
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size  # Returns (width, height)
            return int(width), int(height)  # Ensure integers
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return None, None


def calculate_min_max_dimensions(file_paths: List[str]) -> Dict:
    """
    Calculate min/max width and height for a list of TIFF files.
    
    Args:
        file_paths (List[str]): List of paths to TIFF files
        
    Returns:
        Dict: Dictionary containing min/max dimensions and file info
    """
    widths = []
    heights = []
    valid_files = []
    invalid_files = []
    
    print(f"Processing {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths):
        print(f"Processing {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        width, height = read_tiff_dimensions(file_path)
        
        if width is not None and height is not None:
            # Ensure we have valid integers
            try:
                width = int(width)
                height = int(height)
                widths.append(width)
                heights.append(height)
                valid_files.append({
                    'file': file_path,
                    'width': width,
                    'height': height
                })
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid dimensions for {file_path}: width={width}, height={height}, error={e}")
                invalid_files.append(file_path)
        else:
            invalid_files.append(file_path)
    
    if not widths:
        raise ValueError("No valid TIFF files found!")
    
    min_width = min(widths)
    max_width = max(widths)
    min_height = min(heights)
    max_height = max(heights)
    
    result = {
        'min_width': min_width,
        'max_width': max_width,
        'min_height': min_height,
        'max_height': max_height,
        'width_range': max_width - min_width,
        'height_range': max_height - min_height,
        'total_files_processed': len(file_paths),
        'valid_files': len(valid_files),
        'invalid_files_count': len(invalid_files),
        'valid_file_details': valid_files,
        'invalid_files_list': invalid_files
    }
    
    return result


def save_results(results: Dict, output_dir: str, filename_prefix: str = "imageMinMax"):
    """
    Save results to JSON and text files.
    
    Args:
        results (Dict): Results dictionary
        output_dir (str): Output directory path
        filename_prefix (str): Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{filename_prefix}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {json_path}")
    
    # Save as text file
    txt_path = os.path.join(output_dir, f"{filename_prefix}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Image Min/Max Dimensions Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total files processed: {results['total_files_processed']}\n")
        f.write(f"Valid files: {results['valid_files']}\n")
        f.write(f"Invalid files: {results['invalid_files_count']}\n\n")
        f.write("Dimension Summary:\n")
        f.write(f"  Min Width:  {results['min_width']}\n")
        f.write(f"  Max Width:  {results['max_width']}\n")
        f.write(f"  Width Range (Max-Min): {results['width_range']}\n")
        f.write(f"  Min Height: {results['min_height']}\n")
        f.write(f"  Max Height: {results['max_height']}\n")
        f.write(f"  Height Range (Max-Min): {results['height_range']}\n\n")
        
        if results['invalid_files_count'] > 0:
            f.write("Invalid Files:\n")
            for file_path in results['invalid_files_list']:
                f.write(f"  - {file_path}\n")
            f.write("\n")
        
        f.write("Valid Files Details:\n")
        for file_info in results['valid_file_details']:
            f.write(f"  {os.path.basename(file_info['file'])}: {file_info['width']}x{file_info['height']}\n")
    
    print(f"Text report saved to: {txt_path}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Calculate min/max dimensions of TIFF files')
    parser.add_argument('--files', nargs='+', help='List of TIFF file paths')
    parser.add_argument('--output-dir', default='./output', help='Output directory (default: ./output)')
    parser.add_argument('--filename-prefix', default='imageMinMax', help='Output filename prefix (default: imageMinMax)')
    
    args = parser.parse_args()
    
    # Default file list if not provided
    if not args.files:
        default_files = [
            '/Volumes/T7/20251010_projection/src5/gap/gap_part1to2/0241.tif',
            '/Volumes/T7/20251010_projection/src5/gap/gap_part1to2/0239.tif',
            '/Volumes/T7/20251010_projection/src5/gap/gap_part2to3/0410.tif',
            '/Volumes/T7/20251010_projection/src5/gap/gap_part3to4/0656.tif'
        ]
        print("No files specified, using default file list:")
        for file_path in default_files:
            print(f"  - {file_path}")
        file_paths = default_files
    else:
        file_paths = args.files
    
    try:
        # Calculate min/max dimensions
        results = calculate_min_max_dimensions(file_paths)
        
        # Print summary
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"Total files processed: {results['total_files_processed']}")
        print(f"Valid files: {results['valid_files']}")
        print(f"Invalid files: {results['invalid_files_count']}")
        print(f"Min Width:  {results['min_width']}")
        print(f"Max Width:  {results['max_width']}")
        print(f"Width Range (Max-Min): {results['width_range']}")
        print(f"Min Height: {results['min_height']}")
        print(f"Max Height: {results['max_height']}")
        print(f"Height Range (Max-Min): {results['height_range']}")
        
        # Save results
        save_results(results, args.output_dir, args.filename_prefix)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
