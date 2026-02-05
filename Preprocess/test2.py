import os
from pathlib import Path

def swap_numbers_in_filenames(directory_path):
    """
    读取指定目录中的所有文件，将文件名中的239和303互换
    
    Args:
        directory_path: 目标目录路径
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"错误：目录 {directory_path} 不存在")
        return
    
    # 获取所有文件（不包括子目录）
    files = [f for f in directory.iterdir() if f.is_file()]
    
    if not files:
        print(f"目录 {directory_path} 中没有文件")
        return
    
    # 第一步：将所有包含239的文件重命名为临时名称
    temp_files = []
    for file in files:
        if '239' in file.name:
            temp_name = file.name.replace('239', 'TEMP_NUMBER')
            temp_path = directory / temp_name
            file.rename(temp_path)
            temp_files.append(temp_path)
            print(f"临时重命名: {file.name} -> {temp_name}")
    
    # 第二步：将所有包含303的文件重命名为239
    for file in files:
        if '303' in file.name:
            new_name = file.name.replace('303', '239')
            new_path = directory / new_name
            file.rename(new_path)
            print(f"重命名: {file.name} -> {new_name}")
    
    # 第三步：将所有临时名称的文件重命名为303
    for temp_file in temp_files:
        new_name = temp_file.name.replace('TEMP_NUMBER', '303')
        new_path = directory / new_name
        temp_file.rename(new_path)
        print(f"重命名: {temp_file.name} -> {new_name}")
    
    print(f"\n完成！共处理 {len(files)} 个文件")

if __name__ == "__main__":
    target_directory = "/Volumes/T7/20251204_halfCell/src/code_analysis2/1_align/part4"
    swap_numbers_in_filenames(target_directory)

