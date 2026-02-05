#!/usr/bin/env python3
"""
Script to pad TIF files to maximum dimensions.
Reads TIF files from input directory, finds maximum width and height,
pads smaller images to match maximum dimensions, and stores padding information.
"""

import argparse
import os
import numpy as np
from PIL import Image
import json
import glob
import shutil
import tifffile


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pad TIF files to maximum dimensions')
    parser.add_argument('--input', '-i', 
                        default='/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1',
                       help='Root directory containing multiple part subdirectories with TIF files')
    parser.add_argument('--output_dir', '-o',
                       default='/Volumes/T7/20251112_EMaLM/20251112 293T/2/_processed_data/_legacy/part1_2_pad',
                       help='Output directory for padded TIF files (default: same as input)')
    parser.add_argument('--mode', choices=['auto', 'manual', 'target_size'], default='target_size',
                       help='Padding mode: auto (pad to max dimensions), manual (pad by specific amounts), target_size (pad to specific target size)')
    '''
    会根据--target_width、--target_height和--pad_top, --pad_left 填充量自动计算出右下填充量
    ┌─────────────────────────────────────────────────────────┐
    │ 上填充 (50px)                                            │
    ├─────────────────────────────────────────────────────────┤
    │ 左填充  │              原始图像              │ 右填充      │
    │ (100px)│         (5266×2304)               │ (634px)    │
    │        │                                   │            │
    ├─────────────────────────────────────────────────────────┤
    │ 下填充 (146px)                                           │
    └─────────────────────────────────────────────────────────┘
    mode:
        - target_size: 
        target_width = --target_width # 手动输入
        target_height = --target_height

        - auto: 
        target_width = max_width   # 所有图像中的最大宽度
        target_height = max_height # 所有图像中的最大高度
        
        - manual: 只由--pad_left, --pad_right, --pad_top, --pad_bottom 决定
        target_width = max_width + max(0, args.pad_left) + max(0, args.pad_right)
        target_height = max_height + max(0, args.pad_top) + max(0, args.pad_bottom)
    '''
    parser.add_argument('--target_width', default=6816, type=int, help='Target width in pixels (only used with --mode target_size')
    parser.add_argument('--target_height', default=10827, type=int, help='Target height in pixels (only used with --mode target_size')

    parser.add_argument('--pad_left', type=int, default=150, help='Padding on the left (pixels) - used with manual or target_size mode')
    parser.add_argument('--pad_right', type=int, default=150, help='Padding on the right (pixels) - used with manual or target_size mode')
    parser.add_argument('--pad_top', type=int, default=500, help='Padding on the top (pixels) - used with manual or target_size mode')
    parser.add_argument('--pad_bottom', type=int, default=1500, help='Padding on the bottom (pixels) - used with manual or target_size mode')

    parser.add_argument('--lzw-compression', type=bool, default=True,
                       help='Enable/disable LZW compression for output files (True/False, default: True)')
    
    return parser.parse_args()


def get_image_dimensions(image_path):
    """Get dimensions of a TIF image."""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None


def is_tif_stack(image_path):
    """Check if a TIF file is a stack (multi-layer)."""
    try:
        with Image.open(image_path) as img:
            # Try to seek to the next frame
            img.seek(1)
            return True  # If we can seek to frame 1, it's a stack
    except EOFError:
        return False  # Single frame
    except Exception as e:
        print(f"Error checking stack status for {image_path}: {e}")
        return False


def get_stack_info(image_path):
    """Get information about a TIF stack including dimensions and layer count."""
    try:
        with Image.open(image_path) as img:
            # Get first frame dimensions
            width, height = img.size
            
            # Count total frames
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                pass  # Reached end of frames
            
            return {
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'is_stack': frame_count > 1
            }
    except Exception as e:
        print(f"Error reading stack info for {image_path}: {e}")
        return None


def find_max_dimensions(input_path):
    """Find maximum width and height among TIF files.
    
    Args:
        input_path: Can be a directory path, single TIF file path, or list of paths (directories or TIF files)
    
    Returns:
        tuple: (max_width, max_height, file_dimensions, tif_files, stack_info) or (None, None, None, None, None) if error
    """
    tif_files = []
    file_subdirs = {}
    
    # Handle different input types
    if isinstance(input_path, list):
        # Input is a list of paths (can be directories or TIF files)
        for path in input_path:
            base_name = None
            collected_files = []
            if os.path.isfile(path) and path.lower().endswith('.tif'):
                # It's a TIF file, add it directly
                tif_files.append(path)
                base_name = os.path.splitext(os.path.basename(path))[0]
                collected_files = [path]
            elif os.path.isdir(path):
                # It's a directory, find all TIF files in it
                dir_tif_files = glob.glob(os.path.join(path, "*.tif"))
                tif_files.extend(dir_tif_files)
                collected_files = dir_tif_files
                base_name = os.path.basename(os.path.normpath(path))
                print(f"Found {len(dir_tif_files)} TIF files in directory: {path}")
            else:
                print(f"Warning: Skipping invalid path: {path}")
                continue
            if base_name:
                for file_path in collected_files:
                    file_subdirs[file_path] = base_name
        print(f"Processing list of {len(tif_files)} TIF files from {len(input_path)} input paths...")
    elif os.path.isfile(input_path) and input_path.lower().endswith('.tif'):
        # Input is a single TIF file
        tif_files = [input_path]
        print(f"Processing single TIF file: {os.path.basename(input_path)}")
    elif os.path.isdir(input_path):
        # Input is a directory
        tif_files = glob.glob(os.path.join(input_path, "*.tif"))
        print(f"Processing directory with {len(tif_files)} TIF files...")
    else:
        print(f"Error: Invalid input path {input_path}")
        return None, None, None, None, None
    
    if not tif_files:
        print(f"No TIF files found in {input_path}")
        return None, None, None, None, None
    
    max_width = 0
    max_height = 0
    file_dimensions = {}
    stack_info = {}
    
    print(f"Scanning {len(tif_files)} TIF files for dimensions...")
    
    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        
        # Check if it's a stack
        if is_tif_stack(tif_file):
            stack_data = get_stack_info(tif_file)
            if stack_data:
                width, height = stack_data['width'], stack_data['height']
                file_dimensions[tif_file] = (width, height)
                stack_info[tif_file] = stack_data
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                print(f"{filename}: {width}x{height} (stack with {stack_data['frame_count']} layers)")
        else:
            # Regular single-frame TIF
            dimensions = get_image_dimensions(tif_file)
            if dimensions:
                width, height = dimensions
                file_dimensions[tif_file] = (width, height)
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                print(f"{filename}: {width}x{height}")
    
    print(f"\nMaximum dimensions found: {max_width}x{max_height}")
    return max_width, max_height, file_dimensions, tif_files, stack_info, file_subdirs


def copy_results_files(part_dirs, output_root):
    """Copy each part's Results.csv into the matching output directory."""
    copied = 0
    for part_dir in part_dirs:
        part_name = os.path.basename(os.path.normpath(part_dir))
        src_file = os.path.join(part_dir, "Results.csv")
        if not os.path.isfile(src_file):
            print(f"Warning: Results.csv not found in {part_dir}, skipping.")
            continue
        
        target_dir = os.path.join(output_root, part_name)
        os.makedirs(target_dir, exist_ok=True)
        dst_file = os.path.join(target_dir, "Results.csv")
        
        try:
            shutil.copy2(src_file, dst_file)
            copied += 1
            print(f"Copied {src_file} -> {dst_file}")
        except Exception as exc:
            print(f"Error copying {src_file} to {dst_file}: {exc}")
    
    if copied == 0:
        print("No Results.csv files were copied.")
    else:
        print(f"Copied {copied} Results.csv files.")


def pad_image_automatic(image_path, max_width, max_height, output_path, use_lzw=True):
    """Pad an image to maximum dimensions by adding pixels to the right and bottom (original algorithm)."""
    try:
        with Image.open(image_path) as img:
            current_width, current_height = img.size
            
            # Calculate padding needed
            pad_right = max_width - current_width
            pad_bottom = max_height - current_height
            
            if pad_right <= 0 and pad_bottom <= 0:
                # Image is already at max size or larger
                print(f"  {os.path.basename(image_path)}: No padding needed")
                # Convert PIL image to numpy array for tifffile
                img_array = np.array(img)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, img_array, compression=compression)
                return 0, 0, 0, 0, current_width, current_height
            
            # Create new image with max dimensions
            new_img = Image.new(img.mode, (max_width, max_height), 0)  # Fill with black (0)
            
            # Paste original image at top-left
            new_img.paste(img, (0, 0))
            
            # Convert PIL image to numpy array for tifffile
            new_img_array = np.array(new_img)
            compression = 'lzw' if use_lzw else None
            tifffile.imwrite(output_path, new_img_array, compression=compression)
            
            print(f"  {os.path.basename(image_path)}: Padded {pad_right} right, {pad_bottom} bottom")
            return 0, 0, pad_right, pad_bottom, current_width, current_height
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None, None, None


def pad_image_manual(image_path, target_width, target_height, output_path, pad_left=0, pad_top=0, use_lzw=True):
    """Pad an image to target dimensions with manual padding values.

    The image is pasted at offset (pad_left, pad_top), effectively adding
    the requested left/top margins for all images. The remaining space to
    reach target width/height is filled on the right/bottom.
    """
    try:
        with Image.open(image_path) as img:
            current_width, current_height = img.size
            
            # Calculate padding needed on right/bottom given fixed left/top
            pad_right = max(0, target_width - pad_left - current_width)
            pad_bottom = max(0, target_height - pad_top - current_height)
            
            if pad_left <= 0 and pad_top <= 0 and pad_right <= 0 and pad_bottom <= 0:
                # Image already fits target canvas; just save as is
                print(f"  {os.path.basename(image_path)}: No padding needed")
                img_array = np.array(img)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, img_array, compression=compression)
                return 0, 0, 0, 0, current_width, current_height
            
            # Create new image with target dimensions
            new_img = Image.new(img.mode, (target_width, target_height), 0)  # Fill with black (0)
            
            # Paste original image with the requested left/top offset
            new_img.paste(img, (max(0, pad_left), max(0, pad_top)))
            
            # Convert PIL image to numpy array for tifffile
            new_img_array = np.array(new_img)
            compression = 'lzw' if use_lzw else None
            tifffile.imwrite(output_path, new_img_array, compression=compression)
            
            print(
                f"  {os.path.basename(image_path)}: Padded L{max(0, pad_left)} T{max(0, pad_top)} R{pad_right} B{pad_bottom}"
            )
            return max(0, pad_left), max(0, pad_top), pad_right, pad_bottom, current_width, current_height
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None, None, None


def pad_image_target_size(image_path, target_width, target_height, output_path, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, use_lzw=True):
    """Pad an image to specific target dimensions with fixed padding amounts.
    
    This function pads the image to reach the exact target dimensions by adding
    the specified padding amounts on each side.
    """
    try:
        with Image.open(image_path) as img:
            current_width, current_height = img.size
            
            # Calculate the final dimensions after padding
            final_width = current_width + pad_left + pad_right
            final_height = current_height + pad_top + pad_bottom
            
            # Check if we need to adjust padding to reach target dimensions
            if final_width != target_width or final_height != target_height:
                # Adjust padding to reach exact target dimensions
                pad_right = target_width - current_width - pad_left
                pad_bottom = target_height - current_height - pad_top
                
                # Ensure padding values are non-negative
                if pad_right < 0 or pad_bottom < 0:
                    print(f"  {os.path.basename(image_path)}: Warning - target dimensions too small for current image size")
                    pad_right = max(0, pad_right)
                    pad_bottom = max(0, pad_bottom)
            
            if pad_left <= 0 and pad_top <= 0 and pad_right <= 0 and pad_bottom <= 0:
                # Image already fits target canvas; just save as is
                print(f"  {os.path.basename(image_path)}: No padding needed")
                img_array = np.array(img)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, img_array, compression=compression)
                return 0, 0, 0, 0, current_width, current_height
            
            # Create new image with target dimensions
            new_img = Image.new(img.mode, (target_width, target_height), 0)  # Fill with black (0)
            
            # Paste original image with the requested left/top offset
            new_img.paste(img, (max(0, pad_left), max(0, pad_top)))
            
            # Convert PIL image to numpy array for tifffile
            new_img_array = np.array(new_img)
            compression = 'lzw' if use_lzw else None
            tifffile.imwrite(output_path, new_img_array, compression=compression)
            
            print(
                f"  {os.path.basename(image_path)}: Padded to {target_width}x{target_height} (L{max(0, pad_left)} T{max(0, pad_top)} R{pad_right} B{pad_bottom})"
            )
            return max(0, pad_left), max(0, pad_top), pad_right, pad_bottom, current_width, current_height
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None, None, None


def pad_stack_automatic(stack_path, max_width, max_height, output_path, use_lzw=True):
    """Pad a TIF stack to maximum dimensions by adding pixels to the right and bottom."""
    try:
        with Image.open(stack_path) as img:
            # Get stack info
            stack_info = get_stack_info(stack_path)
            if not stack_info:
                return None, None, None, None, None, None
            
            frame_count = stack_info['frame_count']
            current_width, current_height = stack_info['width'], stack_info['height']
            
            # Calculate padding needed
            pad_right = max_width - current_width
            pad_bottom = max_height - current_height
            
            if pad_right <= 0 and pad_bottom <= 0:
                # Stack is already at max size or larger
                print(f"  {os.path.basename(stack_path)}: No padding needed")
                # Convert to numpy array for tifffile
                frames_array = []
                for frame_idx in range(frame_count):
                    img.seek(frame_idx)
                    frames_array.append(np.array(img))
                frames_array = np.stack(frames_array, axis=0)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, frames_array, compression=compression)
                return 0, 0, 0, 0, current_width, current_height
            
            # Process each frame
            frames_array = []
            for frame_idx in range(frame_count):
                img.seek(frame_idx)
                frame = img.copy()
                
                # Create new frame with max dimensions
                new_frame = Image.new(frame.mode, (max_width, max_height), 0)  # Fill with black (0)
                
                # Paste original frame at top-left
                new_frame.paste(frame, (0, 0))
                frames_array.append(np.array(new_frame))
            
            # Save as multi-page TIFF using tifffile
            if frames_array:
                frames_array = np.stack(frames_array, axis=0)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, frames_array, compression=compression)
                print(f"  {os.path.basename(stack_path)}: Padded {pad_right} right, {pad_bottom} bottom ({frame_count} layers)")
                return 0, 0, pad_right, pad_bottom, current_width, current_height
            
    except Exception as e:
        print(f"Error processing stack {stack_path}: {e}")
        return None, None, None, None, None, None


def pad_stack_manual(stack_path, target_width, target_height, output_path, pad_left=0, pad_top=0, use_lzw=True):
    """Pad a TIF stack to target dimensions with manual padding values."""
    try:
        with Image.open(stack_path) as img:
            # Get stack info
            stack_info = get_stack_info(stack_path)
            if not stack_info:
                return None, None, None, None, None, None
            
            frame_count = stack_info['frame_count']
            current_width, current_height = stack_info['width'], stack_info['height']
            
            # Calculate padding needed on right/bottom given fixed left/top
            pad_right = max(0, target_width - pad_left - current_width)
            pad_bottom = max(0, target_height - pad_top - current_height)
            
            if pad_left <= 0 and pad_top <= 0 and pad_right <= 0 and pad_bottom <= 0:
                # Stack already fits target canvas; just save as is
                print(f"  {os.path.basename(stack_path)}: No padding needed")
                # Convert to numpy array for tifffile
                frames_array = []
                for frame_idx in range(frame_count):
                    img.seek(frame_idx)
                    frames_array.append(np.array(img))
                frames_array = np.stack(frames_array, axis=0)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, frames_array, compression=compression)
                return 0, 0, 0, 0, current_width, current_height
            
            # Process each frame
            frames_array = []
            for frame_idx in range(frame_count):
                img.seek(frame_idx)
                frame = img.copy()
                
                # Create new frame with target dimensions
                new_frame = Image.new(frame.mode, (target_width, target_height), 0)  # Fill with black (0)
                
                # Paste original frame with the requested left/top offset
                new_frame.paste(frame, (max(0, pad_left), max(0, pad_top)))
                frames_array.append(np.array(new_frame))
            
            # Save as multi-page TIFF using tifffile
            if frames_array:
                frames_array = np.stack(frames_array, axis=0)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, frames_array, compression=compression)
                print(f"  {os.path.basename(stack_path)}: Padded L{max(0, pad_left)} T{max(0, pad_top)} R{pad_right} B{pad_bottom} ({frame_count} layers)")
                return max(0, pad_left), max(0, pad_top), pad_right, pad_bottom, current_width, current_height
            
    except Exception as e:
        print(f"Error processing stack {stack_path}: {e}")
        return None, None, None, None, None, None


def pad_stack_target_size(stack_path, target_width, target_height, output_path, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, use_lzw=True):
    """Pad a TIF stack to specific target dimensions with fixed padding amounts."""
    try:
        with Image.open(stack_path) as img:
            # Get stack info
            stack_info = get_stack_info(stack_path)
            if not stack_info:
                return None, None, None, None, None, None
            
            frame_count = stack_info['frame_count']
            current_width, current_height = stack_info['width'], stack_info['height']
            
            # Calculate the final dimensions after padding
            final_width = current_width + pad_left + pad_right
            final_height = current_height + pad_top + pad_bottom
            
            # Check if we need to adjust padding to reach target dimensions
            if final_width != target_width or final_height != target_height:
                # Adjust padding to reach exact target dimensions
                pad_right = target_width - current_width - pad_left
                pad_bottom = target_height - current_height - pad_top
                
                # Ensure padding values are non-negative
                if pad_right < 0 or pad_bottom < 0:
                    print(f"  {os.path.basename(stack_path)}: Warning - target dimensions too small for current stack size")
                    pad_right = max(0, pad_right)
                    pad_bottom = max(0, pad_bottom)
            
            if pad_left <= 0 and pad_top <= 0 and pad_right <= 0 and pad_bottom <= 0:
                # Stack already fits target canvas; just save as is
                print(f"  {os.path.basename(stack_path)}: No padding needed")
                # Convert to numpy array for tifffile
                frames_array = []
                for frame_idx in range(frame_count):
                    img.seek(frame_idx)
                    frames_array.append(np.array(img))
                frames_array = np.stack(frames_array, axis=0)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, frames_array, compression=compression)
                return 0, 0, 0, 0, current_width, current_height
            
            # Process each frame
            frames_array = []
            for frame_idx in range(frame_count):
                img.seek(frame_idx)
                frame = img.copy()
                
                # Create new frame with target dimensions
                new_frame = Image.new(frame.mode, (target_width, target_height), 0)  # Fill with black (0)
                
                # Paste original frame with the requested left/top offset
                new_frame.paste(frame, (max(0, pad_left), max(0, pad_top)))
                frames_array.append(np.array(new_frame))
            
            # Save as multi-page TIFF using tifffile
            if frames_array:
                frames_array = np.stack(frames_array, axis=0)
                compression = 'lzw' if use_lzw else None
                tifffile.imwrite(output_path, frames_array, compression=compression)
                print(f"  {os.path.basename(stack_path)}: Padded to {target_width}x{target_height} (L{max(0, pad_left)} T{max(0, pad_top)} R{pad_right} B{pad_bottom}) ({frame_count} layers)")
                return max(0, pad_left), max(0, pad_top), pad_right, pad_bottom, current_width, current_height
            
    except Exception as e:
        print(f"Error processing stack {stack_path}: {e}")
        return None, None, None, None, None, None





def main():
    args = parse_arguments()
    
    # Parse input root directory and collect part subdirectories
    input_root = args.input
    if not os.path.exists(input_root):
        print(f"Error: Input root directory {input_root} does not exist")
        return
    if not os.path.isdir(input_root):
        print(f"Error: --input must be a directory containing part folders, got {input_root}")
        return
    
    part_dirs = sorted(
        [
            os.path.join(input_root, name)
            for name in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, name))
        ]
    )
    
    if not part_dirs:
        print(f"Error: No part subdirectories found in {input_root}")
        return
    
    input_path = part_dirs
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle LZW compression flag
    use_lzw = args.lzw_compression
    
    print(f"Input root: {input_root}")
    print(f"Detected {len(input_path)} part directories:")
    for part_dir in input_path:
        print(f"  - {part_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"LZW compression: {'Enabled' if use_lzw else 'Disabled'}")
    print("-" * 50)
    
    # Copy Results.csv files into corresponding output directories
    copy_results_files(input_path, args.output_dir)
    
    # Find maximum dimensions
    result = find_max_dimensions(input_path)
    if result[0] is None:
        return
    
    max_width, max_height, file_dimensions, tif_files, stack_info, file_subdirs = result

    # Determine target dimensions based on mode
    if args.mode == 'target_size':
        # Target size mode: pad to specific dimensions
        if args.target_width is None or args.target_height is None:
            print("Error: --target_width and --target_height are required for target_size mode")
            return
        target_width = args.target_width
        target_height = args.target_height
        print(f"Target size mode: Target dimensions {target_width}x{target_height}")
        print(f"Fixed padding: L{args.pad_left} R{args.pad_right} T{args.pad_top} B{args.pad_bottom}")
    elif args.mode == 'manual':
        # Manual mode: pad by specific amounts around maximum content
        target_width = max_width + max(0, args.pad_left) + max(0, args.pad_right)
        target_height = max_height + max(0, args.pad_top) + max(0, args.pad_bottom)
        print(f"Manual padding mode: Target canvas {target_width}x{target_height}")
        print(f"Manual padding: L{args.pad_left} R{args.pad_right} T{args.pad_top} B{args.pad_bottom}")
    else:  # auto mode
        # Use original automatic mode
        target_width = max_width
        target_height = max_height
        print(f"Automatic padding mode: Target dimensions {target_width}x{target_height}")
    
    # Process each TIF file
    padding_info = {}
    
    print(f"\nProcessing {len(tif_files)} files...")
    print("-" * 50)
    
    list_input = isinstance(input_path, list)
    
    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        target_dir = args.output_dir
        
        if list_input and tif_file in file_subdirs:
            target_dir = os.path.join(args.output_dir, file_subdirs[tif_file])
            os.makedirs(target_dir, exist_ok=True)
        
        if list_input:
            output_filename = f"{os.path.splitext(filename)[0]}_pad.tif"
        else:
            output_filename = filename
        
        output_path = os.path.join(target_dir, output_filename)
        
        print(f"Processing: {filename}")
        
        # Check if it's a stack
        if tif_file in stack_info:
            # Process as TIF stack
            if args.mode == 'target_size':
                p_left, p_top, p_right, p_bottom, cur_w, cur_h = pad_stack_target_size(
                    tif_file, target_width, target_height, output_path, 
                    pad_left=args.pad_left, pad_top=args.pad_top, 
                    pad_right=args.pad_right, pad_bottom=args.pad_bottom, use_lzw=use_lzw
                )
            elif args.mode == 'manual':
                p_left, p_top, p_right, p_bottom, cur_w, cur_h = pad_stack_manual(
                    tif_file, target_width, target_height, output_path, pad_left=args.pad_left, pad_top=args.pad_top, use_lzw=use_lzw
                )
            else:  # auto mode
                p_left, p_top, p_right, p_bottom, cur_w, cur_h = pad_stack_automatic(
                    tif_file, target_width, target_height, output_path, use_lzw=use_lzw
                )
        else:
            # Process as regular single-frame TIF
            if args.mode == 'target_size':
                p_left, p_top, p_right, p_bottom, cur_w, cur_h = pad_image_target_size(
                    tif_file, target_width, target_height, output_path, 
                    pad_left=args.pad_left, pad_top=args.pad_top, 
                    pad_right=args.pad_right, pad_bottom=args.pad_bottom, use_lzw=use_lzw
                )
            elif args.mode == 'manual':
                p_left, p_top, p_right, p_bottom, cur_w, cur_h = pad_image_manual(
                    tif_file, target_width, target_height, output_path, pad_left=args.pad_left, pad_top=args.pad_top, use_lzw=use_lzw
                )
            else:  # auto mode
                p_left, p_top, p_right, p_bottom, cur_w, cur_h = pad_image_automatic(
                    tif_file, target_width, target_height, output_path, use_lzw=use_lzw
                )
        
        if p_left is not None:
            total_padded_pixels = target_width * target_height - cur_w * cur_h
            file_info = {
                'source_file': tif_file,
                'output_file': output_path,
                'original_dimensions': file_dimensions[tif_file],
                'max_content_dimensions': (max_width, max_height),
                'target_canvas_dimensions': (target_width, target_height),
                'padding_mode': args.mode,
                'padding_left': p_left,
                'padding_top': p_top,
                'padding_right': p_right,
                'padding_bottom': p_bottom,
                'total_padding_pixels': total_padded_pixels
            }
            
            # Add stack information if it's a stack
            if tif_file in stack_info:
                file_info['is_stack'] = True
                file_info['stack_info'] = stack_info[tif_file]
            else:
                file_info['is_stack'] = False
            
            padding_info[output_filename] = file_info
    
    # Save padding information
    padding_info_path = os.path.join(args.output_dir, 'padding_info.json')
    with open(padding_info_path, 'w') as f:
        json.dump({
            'max_dimensions': (max_width, max_height),
            'target_dimensions': (target_width, target_height),
            'padding_mode': args.mode,
            'lzw_compression': use_lzw,
            'padding_settings': {
                'pad_left': args.pad_left,
                'pad_right': args.pad_right,
                'pad_top': args.pad_top,
                'pad_bottom': args.pad_bottom
            },
            'target_size_settings': {
                'target_width': args.target_width,
                'target_height': args.target_height
            } if args.mode == 'target_size' else None,
            'files_processed': len(tif_files),
            'padding_details': padding_info
        }, f, indent=2)
    
    print(f"\nPadding information saved to: {padding_info_path}")
    print(f"All files processed and saved to: {args.output_dir}")
    
    # Print summary
    print("\nSummary:")
    print(f"Padding mode: {args.mode}")
    print(f"Maximum content dimensions: {max_width}x{max_height}")
    print(f"Target canvas dimensions: {target_width}x{target_height}")
    if args.mode == 'target_size':
        print(f"Target size: {args.target_width}x{args.target_height}")
        print(f"Fixed padding: L{args.pad_left} R{args.pad_right} T{args.pad_top} B{args.pad_bottom}")
    elif args.mode == 'manual':
        print(f"Manual padding: L{args.pad_left} R{args.pad_right} T{args.pad_top} B{args.pad_bottom}")
    print(f"Files processed: {len(tif_files)}")
    
    total_padded = sum(
        1 for info in padding_info.values()
        if info['padding_left'] > 0 or info['padding_top'] > 0 or info['padding_right'] > 0 or info['padding_bottom'] > 0
    )
    print(f"Files that needed padding: {total_padded}")


if __name__ == "__main__":
    main()
