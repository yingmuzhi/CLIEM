#!/usr/bin/env python3
"""
抽取切片脚本
从输入目录中按顺序每隔n张抽取一张图片，保存到输出目录
"""

import argparse
import os
from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count, Manager


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='抽取切片脚本：从输入目录中按顺序每隔n张抽取一张图片',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_path', '-i', type=str,
                       default='/Volumes/DATA5/src/5_reconstruct_crop',
                       help='输入路径，包含要处理的图片')
    parser.add_argument('--output_path', '-o', type=str,
                       default='/Volumes/DATA5/src/5_reconstruct_crop_extract',
                       help='输出路径，保存抽取后的图片')
    parser.add_argument('--n', '-n', type=int, default=3,
                       help='每隔n张抽取一张（例如：n=2表示每隔2张抽取1张，即抽取第1,3,5...张）')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='并行处理的进程数（默认：CPU核心数）')
    
    return parser.parse_args()


def get_image_files(input_path):
    """获取输入目录下的所有图片文件（排除隐藏文件）"""
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        raise ValueError(f"输入路径不存在: {input_path}")
    
    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录: {input_path}")
    
    # 支持的图片格式
    image_extensions = ['.tif', '.TIF', '.tiff', '.TIFF', '.png', '.PNG', 
                       '.jpg', '.JPG', '.jpeg', '.JPEG']
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
    
    # 过滤掉以点开头的文件（隐藏文件）
    image_files = [f for f in image_files if not f.name.startswith('.')]
    
    # 按文件名排序
    return sorted(image_files)


def get_extract_list(image_files, n):
    """在主进程中确定要抽取的图片列表"""
    # 每隔n张抽取一张（索引从0开始，所以是0, n, 2n, 3n...）
    extract_list = []
    for i in range(0, len(image_files), n):
        extract_list.append(image_files[i])
    return extract_list


def process_single_image(args_tuple):
    """处理单张图片的包装函数，用于多进程调用"""
    source_file, output_dir, index, total, print_lock = args_tuple
    
    try:
        # 生成输出文件路径
        dest_file = output_dir / source_file.name
        
        # 复制文件
        shutil.copy2(source_file, dest_file)
        
        # 使用锁保护打印输出
        with print_lock:
            print(f"[{index}/{total}] 成功: {source_file.name}")
        
        return (True, source_file.name, None)
    except Exception as e:
        # 使用锁保护打印输出
        with print_lock:
            print(f"[{index}/{total}] 失败: {source_file.name} - {str(e)}")
        return (False, source_file.name, str(e))


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        input_path = args.input_path
        output_path = args.output_path
        n = args.n
        
        # 验证n的值
        if n <= 0:
            raise ValueError(f"n必须大于0，当前值: {n}")
        
        # 创建输出目录
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_files = get_image_files(input_path)
        
        if not image_files:
            print(f"在 {input_path} 中未找到图片文件")
            return
        
        print("=" * 50)
        print("抽取切片工具（多进程版本）")
        print("=" * 50)
        print(f"\n找到 {len(image_files)} 张图片")
        print(f"输入目录: {input_path}")
        print(f"输出目录: {output_path}")
        print(f"抽取间隔: 每隔 {n} 张抽取 1 张")
        
        # 在主进程中确定要抽取的图片列表
        extract_list = get_extract_list(image_files, n)
        num_extracted = len(extract_list)
        print(f"将抽取 {num_extracted} 张图片")
        
        # 确定进程数
        num_processes = args.num_processes if args.num_processes else cpu_count()
        print(f"使用 {num_processes} 个进程进行并行处理\n")
        
        # 创建共享锁用于保护打印输出
        manager = Manager()
        print_lock = manager.Lock()
        
        # 准备多进程参数
        process_args = [
            (source_file, output_dir, i+1, num_extracted, print_lock)
            for i, source_file in enumerate(extract_list)
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
        print(f"总共: {len(image_files)} 张")
        print(f"抽取: {num_extracted} 张")
        print(f"成功: {success_count} 张")
        print(f"失败: {fail_count} 张")
        print(f"输出目录: {output_path}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

