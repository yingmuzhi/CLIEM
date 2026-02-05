'''
Author: yingmuzhi cyxscj@126.com
Date: 2025-10-21 21:35:48
LastEditors: yingmuzhi cyxscj@126.com
LastEditTime: 2025-10-22 11:11:28
FilePath: /20251010_projection/code/calculate_resultCSVMinMax.py
Description: 处理多个Results.csv文件，计算dX和dY的最大最小值
'''

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple

def process_results_csv_files(csv_files: List[str]) -> Dict:
    """
    处理多个Results.csv文件，计算dX和dY的最大最小值
    
    Args:
        csv_files: Results.csv文件路径列表
        
    Returns:
        包含每个文件和总体统计信息的字典
    """
    results = {
        'file_stats': [],
        'overall_stats': {
            'dX_min': float('inf'),
            'dX_max': float('-inf'),
            'dY_min': float('inf'),
            'dY_max': float('-inf')
        }
    }
    
    print("开始处理Results.csv文件...")
    print("=" * 50)
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"处理文件 {i}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否包含dX和dY列
            if 'dX' not in df.columns or 'dY' not in df.columns:
                print(f"警告: 文件 {csv_file} 不包含dX或dY列")
                continue
            
            # 计算当前文件的统计信息
            dX_min = df['dX'].min()
            dX_max = df['dX'].max()
            dY_min = df['dY'].min()
            dY_max = df['dY'].max()
            
            file_stats = {
                'file_path': csv_file,
                'file_name': os.path.basename(csv_file),
                'dX_min': dX_min,
                'dX_max': dX_max,
                'dY_min': dY_min,
                'dY_max': dY_max,
                'total_rows': len(df)
            }
            
            results['file_stats'].append(file_stats)
            
            # 更新总体统计信息
            results['overall_stats']['dX_min'] = min(results['overall_stats']['dX_min'], dX_min)
            results['overall_stats']['dX_max'] = max(results['overall_stats']['dX_max'], dX_max)
            results['overall_stats']['dY_min'] = min(results['overall_stats']['dY_min'], dY_min)
            results['overall_stats']['dY_max'] = max(results['overall_stats']['dY_max'], dY_max)
            
            print(f"  dX范围: [{dX_min}, {dX_max}]")
            print(f"  dY范围: [{dY_min}, {dY_max}]")
            print(f"  数据行数: {len(df)}")
            print()
            
        except Exception as e:
            print(f"错误: 无法处理文件 {csv_file}: {str(e)}")
            print()
            continue
    
    return results

def print_results(results: Dict):
    """打印处理结果"""
    print("=" * 50)
    print("处理结果汇总")
    print("=" * 50)
    
    # 打印每个文件的统计信息
    print("\n各文件统计信息:")
    print("-" * 30)
    for i, file_stat in enumerate(results['file_stats'], 1):
        print(f"{i}. {file_stat['file_name']}")
        print(f"   dX范围: [{file_stat['dX_min']}, {file_stat['dX_max']}]")
        print(f"   dY范围: [{file_stat['dY_min']}, {file_stat['dY_max']}]")
        print(f"   数据行数: {file_stat['total_rows']}")
        print()
    
    # 打印总体统计信息
    overall = results['overall_stats']
    print("总体统计信息:")
    print("-" * 30)
    print(f"dX最小值: {overall['dX_min']}")
    print(f"dX最大值: {overall['dX_max']}")
    print(f"dY最小值: {overall['dY_min']}")
    print(f"dY最大值: {overall['dY_max']}")
    print()
    
    # 存储四个关键值
    key_values = {
        'dX_min': overall['dX_min'],
        'dX_max': overall['dX_max'],
        'dY_min': overall['dY_min'],
        'dY_max': overall['dY_max']
    }
    
    print("关键值存储:")
    print("-" * 30)
    for key, value in key_values.items():
        print(f"{key}: {value}")
    
    return key_values

def main():
    """主函数"""
    # 示例文件路径列表
    csv_files = [
        '/Volumes/T7/20251010_projection/src5/1_align/part1/Results.csv',
        '/Volumes/T7/20251010_projection/src5/1_align/part2/Results.csv',
        '/Volumes/T7/20251010_projection/src5/1_align/part3/Results.csv',
        '/Volumes/T7/20251010_projection/src5/1_align/part4/Results.csv'
    ]
    
    # 检查文件是否存在
    existing_files = []
    for file_path in csv_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"警告: 文件不存在 {file_path}")
    
    if not existing_files:
        print("错误: 没有找到任何有效的Results.csv文件")
        return
    
    # 处理文件
    results = process_results_csv_files(existing_files)
    
    if not results['file_stats']:
        print("错误: 没有成功处理任何文件")
        return
    
    # 打印结果
    key_values = print_results(results)
    
    print("\n处理完成!")

if __name__ == "__main__":
    main()
