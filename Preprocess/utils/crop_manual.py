#!/usr/bin/env python3
"""
手动裁剪图片脚本
从命令行参数获取裁剪参数，批量处理指定目录下的所有图片
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import tifffile
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='手动裁剪图片脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--crop_top', '-t', type=int, default=6100,
                       help='顶部裁剪像素值')
    parser.add_argument('--crop_bottom', '-b', type=int, default=3000,
                       help='底部裁剪像素值')
    parser.add_argument('--crop_left', '-l', type=int, default=0,
                       help='左侧裁剪像素值')
    parser.add_argument('--crop_right', '-r', type=int, default=0,
                       help='右侧裁剪像素值')
    parser.add_argument('--input_path', '-i', type=str, 
                       default='/Volumes/DATA5/src/5_reconstruct',
                       help='输入路径，包含要处理的图片')
    parser.add_argument('--output_path', '-o', type=str, 
                       default='/Volumes/DATA5/src/5_reconstruct_crop',
                       help='输出路径，保存裁剪后的图片')
    parser.add_argument('--num_processes', '-n', type=int, default=10,
                       help='并行处理的进程数（默认：CPU核心数）')
    
    return parser.parse_args()


def get_image_files(input_path):
    """获取输入目录下的所有TIF文件（排除隐藏文件）"""
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        raise ValueError(f"输入路径不存在: {input_path}")
    
    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录: {input_path}")
    
    # 只读取以 .tif 或 .TIF 结尾的文件，且不以点开头（排除隐藏文件）
    image_files = []
    image_files.extend(input_dir.glob("*.tif"))
    image_files.extend(input_dir.glob("*.TIF"))
    
    # 过滤掉以点开头的文件（隐藏文件）
    image_files = [f for f in image_files if not f.name.startswith('.')]
    
    return sorted(image_files)


def crop_image(image_path, crop_top, crop_bottom, crop_left, crop_right, output_path):
    """裁剪单张图片"""
    image_path = Path(image_path)
    
    # 判断是否为TIF文件（可能需要特殊处理）
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            # 尝试使用tifffile读取（支持多页TIF）
            img_array = tifffile.imread(str(image_path))
            
            # 如果是2D数组，直接处理
            if len(img_array.shape) == 2:
                h, w = img_array.shape
                cropped = img_array[crop_top:h-crop_bottom, crop_left:w-crop_right]
                tifffile.imwrite(str(output_path), cropped)
            # 如果是3D数组（多页TIF），逐页处理
            elif len(img_array.shape) == 3:
                h, w = img_array.shape[1], img_array.shape[2]
                cropped = img_array[:, crop_top:h-crop_bottom, crop_left:w-crop_right]
                tifffile.imwrite(str(output_path), cropped)
            else:
                raise ValueError(f"不支持的图像维度: {img_array.shape}")
                
        except Exception as e:
            # 如果tifffile读取失败，尝试使用PIL
            print(f"使用tifffile读取失败，尝试使用PIL: {e}")
            img = Image.open(image_path)
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            cropped_array = img_array[crop_top:h-crop_bottom, crop_left:w-crop_right]
            cropped_img = Image.fromarray(cropped_array)
            cropped_img.save(output_path)
    else:
        # 使用PIL处理其他格式
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 处理RGB/RGBA图像
        if len(img_array.shape) == 3:
            h, w = img_array.shape[:2]
            cropped_array = img_array[crop_top:h-crop_bottom, crop_left:w-crop_right]
        # 处理灰度图像
        elif len(img_array.shape) == 2:
            h, w = img_array.shape
            cropped_array = img_array[crop_top:h-crop_bottom, crop_left:w-crop_right]
        else:
            raise ValueError(f"不支持的图像维度: {img_array.shape}")
        
        cropped_img = Image.fromarray(cropped_array)
        cropped_img.save(output_path)


def process_single_image(args_tuple):
    """处理单张图片的包装函数，用于多进程调用"""
    img_path, crop_top, crop_bottom, crop_left, crop_right, output_dir, index, total, print_lock = args_tuple
    
    try:
        # 生成输出文件路径
        output_file = output_dir / img_path.name
        
        # 裁剪图片
        crop_image(img_path, crop_top, crop_bottom, crop_left, crop_right, output_file)
        
        # 使用锁保护打印输出
        with print_lock:
            print(f"[{index}/{total}] 成功: {img_path.name}")
        
        return (True, img_path.name, None)
    except Exception as e:
        # 使用锁保护打印输出
        with print_lock:
            print(f"[{index}/{total}] 失败: {img_path.name} - {str(e)}")
        return (False, img_path.name, str(e))


def validate_crop_params(image_path, crop_top, crop_bottom, crop_left, crop_right):
    """验证裁剪参数是否有效"""
    # 读取图片尺寸
    if Path(image_path).suffix.lower() in ['.tif', '.tiff']:
        try:
            img_array = tifffile.imread(str(image_path))
            if len(img_array.shape) == 2:
                h, w = img_array.shape
            elif len(img_array.shape) == 3:
                h, w = img_array.shape[1], img_array.shape[2]
            else:
                return False, f"不支持的图像维度: {img_array.shape}"
        except:
            img = Image.open(image_path)
            h, w = img.size[1], img.size[0]
    else:
        img = Image.open(image_path)
        h, w = img.size[1], img.size[0]
    
    # 检查裁剪参数
    if crop_top < 0 or crop_bottom < 0 or crop_left < 0 or crop_right < 0:
        return False, "裁剪像素值不能为负数"
    
    if crop_top + crop_bottom >= h:
        return False, f"顶部和底部裁剪像素值之和 ({crop_top + crop_bottom}) 不能大于等于图像高度 ({h})"
    
    if crop_left + crop_right >= w:
        return False, f"左侧和右侧裁剪像素值之和 ({crop_left + crop_right}) 不能大于等于图像宽度 ({w})"
    
    return True, (h, w)


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        crop_top = args.crop_top
        crop_bottom = args.crop_bottom
        crop_left = args.crop_left
        crop_right = args.crop_right
        input_path = args.input_path
        output_path = args.output_path
        
        # 创建输出目录
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_files = get_image_files(input_path)
        
        if not image_files:
            print(f"在 {input_path} 中未找到图片文件")
            return
        
        print("=" * 50)
        print("图片裁剪工具（多进程版本）")
        print("=" * 50)
        print(f"\n找到 {len(image_files)} 张图片")
        print(f"输出目录: {output_path}")
        print(f"裁剪参数: top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right}")
        
        # 验证裁剪参数（使用第一张图片验证）
        if image_files:
            is_valid, result = validate_crop_params(image_files[0], crop_top, crop_bottom, crop_left, crop_right)
            if not is_valid:
                print(f"错误: {result}")
                return
            print(f"图像尺寸: {result[1]} x {result[0]}")
            print(f"裁剪后尺寸: {result[1] - crop_left - crop_right} x {result[0] - crop_top - crop_bottom}")
        
        # 确定进程数
        num_processes = args.num_processes if args.num_processes else cpu_count()
        print(f"使用 {num_processes} 个进程进行并行处理\n")
        
        # 创建共享锁用于保护打印输出
        manager = Manager()
        print_lock = manager.Lock()
        
        # 准备多进程参数
        process_args = [
            (img_path, crop_top, crop_bottom, crop_left, crop_right, output_dir, i+1, len(image_files), print_lock)
            for i, img_path in enumerate(image_files)
        ]
        
        # 使用多进程处理
        success_count = 0
        fail_count = 0
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_single_image, process_args)
        
        # 统计结果
        for success, filename, error in results:
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        # 输出统计信息
        print("\n" + "=" * 50)
        print(f"处理完成!")
        print(f"成功: {success_count} 张")
        print(f"失败: {fail_count} 张")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

