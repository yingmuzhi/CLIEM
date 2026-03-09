import pandas as pd
import numpy as np
import tifffile
import os
import glob
from pathlib import Path

# 使用pandas读入localMatrix.txt文件
file_path = '/Volumes/T7/20251010_halfCell/20251119_amira_o/1_amira_output/localMatrix.txt'
df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-16le')

print(f"成功读取文件，共 {len(df)} 行数据")
print("\n数据预览：")
print(df.head())

# 原图大小
ORIGINAL_X = 4663
ORIGINAL_Y = 4315
ORIGINAL_SLICE = 425

# 输出目录
output_base_dir = '/Volumes/T7/20251010_halfCell/20251119_amira_o/output'
os.makedirs(output_base_dir, exist_ok=True)

# 处理每一行数据
for idx, row in df.iterrows():
    # 第0列是数据保存位置（目录路径）
    data_dir = row[0]
    # 第1-3列是起始位置(x, y, slice)
    start_x = int(row[1])
    start_y = int(row[2])
    start_slice = int(row[3])
    # 第4-6列是crop的体积(x, y, slice总数)
    crop_x = int(row[4])
    crop_y = int(row[5])
    crop_slice = int(row[6])
    
    print(f"\n处理第 {idx+1} 组数据:")
    print(f"  数据目录: {data_dir}")
    print(f"  起始位置: x={start_x}, y={start_y}, slice={start_slice}")
    print(f"  Crop大小: x={crop_x}, y={crop_y}, slice={crop_slice}")
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"  警告: 目录不存在，跳过: {data_dir}")
        continue
    
    # 查找目录下的所有tif文件
    tif_pattern = os.path.join(data_dir, '*.tif')
    all_tif_files = sorted(glob.glob(tif_pattern))
    
    # 过滤掉.info文件（如果有的话）
    all_tif_files = [f for f in all_tif_files if not f.endswith('.info')]
    
    if len(all_tif_files) == 0:
        print(f"  警告: 目录下没有找到tif文件: {data_dir}")
        continue
    
    print(f"  找到 {len(all_tif_files)} 个tif文件")
    
    # 根据文件数量决定如何选择文件
    if len(all_tif_files) == crop_slice:
        # 文件数量正好等于crop_slice，使用所有文件
        tif_files = all_tif_files
        print(f"  使用所有 {len(tif_files)} 个文件")
    elif len(all_tif_files) > crop_slice:
        # 文件数量大于crop_slice，根据start_slice选择文件
        # 假设文件编号对应slice编号（如000对应slice 0）
        # 从start_slice对应的文件开始，读取crop_slice个文件
        start_file_idx = start_slice
        end_file_idx = start_slice + crop_slice
        if end_file_idx > len(all_tif_files):
            print(f"  警告: 需要的文件范围({start_file_idx}-{end_file_idx})超出可用文件数量({len(all_tif_files)})")
            end_file_idx = len(all_tif_files)
        tif_files = all_tif_files[start_file_idx:end_file_idx]
        print(f"  从文件索引 {start_file_idx} 开始，使用 {len(tif_files)} 个文件")
    else:
        # 文件数量小于crop_slice，使用所有可用文件
        tif_files = all_tif_files
        print(f"  警告: tif文件数量({len(tif_files)})小于crop_slice({crop_slice})，使用所有可用文件")
    
    # 创建原图大小的全零数组
    full_image = np.zeros((ORIGINAL_SLICE, ORIGINAL_Y, ORIGINAL_X), dtype=np.uint8)
    
    # 读取并放置每个tif文件
    for i, tif_file in enumerate(tif_files):
        try:
            # 读取tif文件
            img = tifffile.imread(tif_file)
            img_height, img_width = img.shape
            
            # 判断文件是完整大小的还是已经裁剪的
            is_full_size = (img_height == ORIGINAL_Y and img_width == ORIGINAL_X)
            is_cropped = (img_height == crop_y and img_width == crop_x)
            
            if is_cropped:
                # 文件已经是裁剪后的，直接使用
                img_crop = img
            elif is_full_size:
                # 文件是完整大小的，需要从中提取crop区域
                crop_start_y = start_y
                crop_start_x = start_x
                crop_end_y = start_y + crop_y
                crop_end_x = start_x + crop_x
                
                # 检查crop区域是否在图像范围内
                if crop_start_y < 0 or crop_start_x < 0:
                    print(f"  警告: crop起始位置超出范围，跳过文件 {os.path.basename(tif_file)}")
                    continue
                
                # 如果crop区域超出图像边界，进行裁剪
                if crop_end_y > img_height:
                    crop_end_y = img_height
                if crop_end_x > img_width:
                    crop_end_x = img_width
                
                # 提取crop区域
                img_crop = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            else:
                # 文件尺寸不匹配，尝试自动处理
                print(f"  警告: 文件 {os.path.basename(tif_file)} 的尺寸 {img.shape} 既不是完整大小({ORIGINAL_Y}, {ORIGINAL_X})也不是crop大小({crop_y}, {crop_x})")
                # 如果图像尺寸大于crop尺寸，尝试裁剪
                if img_height >= crop_y and img_width >= crop_x:
                    img_crop = img[:crop_y, :crop_x]
                else:
                    # 如果图像较小，创建正确大小的数组并填充
                    img_crop = np.zeros((crop_y, crop_x), dtype=img.dtype)
                    actual_h = min(img_height, crop_y)
                    actual_w = min(img_width, crop_x)
                    img_crop[:actual_h, :actual_w] = img[:actual_h, :actual_w]
            
            # 计算在完整图像中的位置
            current_slice = start_slice + i
            
            # 检查边界
            if current_slice >= ORIGINAL_SLICE:
                print(f"  警告: slice索引 {current_slice} 超出范围，跳过")
                continue
            
            # 计算在输出图像中的目标位置
            target_start_y = start_y
            target_start_x = start_x
            target_end_y = start_y + img_crop.shape[0]
            target_end_x = start_x + img_crop.shape[1]
            
            # 检查目标位置是否超出输出图像边界
            if target_end_y > ORIGINAL_Y:
                target_end_y = ORIGINAL_Y
                img_crop = img_crop[:target_end_y - target_start_y, :]
            if target_end_x > ORIGINAL_X:
                target_end_x = ORIGINAL_X
                img_crop = img_crop[:, :target_end_x - target_start_x]
            
            # 将crop区域放置到完整图像中
            full_image[current_slice, target_start_y:target_end_y, target_start_x:target_end_x] = img_crop
            
            if (i + 1) % 50 == 0:
                print(f"    已处理 {i+1}/{len(tif_files)} 个文件")
        
        except Exception as e:
            print(f"  错误: 读取文件 {tif_file} 时出错: {e}")
            continue
    
    # 创建输出目录（使用目录名作为文件夹名）
    # 获取相对于基础目录的路径部分
    base_path = '/Volumes/T7/20251010_halfCell/20251119_amira_o'
    if data_dir.startswith(base_path):
        # 获取相对路径部分
        rel_path = data_dir[len(base_path):].lstrip('/').rstrip('/')
        # 将路径中的斜杠替换为下划线作为目录名
        dir_name = rel_path.replace('/', '_') if rel_path else 'output'
    else:
        # 如果不在基础路径下，使用目录名
        dir_name = os.path.basename(data_dir.rstrip('/'))
        if not dir_name:
            dir_name = f"group_{idx+1}"
    
    output_dir = os.path.join(output_base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果：将3D数组保存为多个2D tif文件
    print(f"  保存结果到: {output_dir}")
    for slice_idx in range(ORIGINAL_SLICE):
        output_file = os.path.join(output_dir, f'slice_{slice_idx:04d}.tif')
        tifffile.imwrite(output_file, full_image[slice_idx])
    
    print(f"  完成！共保存 {ORIGINAL_SLICE} 个slice文件")

print("\n所有处理完成！")
