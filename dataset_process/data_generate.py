import os
import shutil
from pathlib import Path

# 定义源文件夹和目标文件夹
source_folder = Path('C:/Users/guozy/Desktop/data/dataset_final_patient')
txt_file_path = Path('C:/Users/guozy/Desktop/data/non_collapse.txt')  # 替换为你的txt文件路径
target_folder = Path('C:/Users/guozy/Desktop/data/non_collapse')  # 目标文件夹，存储复制的图片

# 确保目标文件夹存在
if not target_folder.exists():
    target_folder.mkdir(parents=True)

# 读取txt文件中的子文件夹名
with open(txt_file_path, 'r') as file:
    subfolders = file.read().splitlines()

# 从1000开始编号
start_index = 2000

for subfolder_name in subfolders:
    # 构造原子文件夹路径
    original_subfolder_path = source_folder / subfolder_name
    coronal_folder_path = original_subfolder_path / 'Coronal'

    # 确保Coronal文件夹存在
    if coronal_folder_path.exists():
        print(subfolder_name)
        for i in range(75):
            # 构造原图片文件路径
            original_image_name = f"image_000{str(i).zfill(2)}.png"  # 假设图片格式为png
            original_image_path = coronal_folder_path / original_image_name

            # 构造新图片文件名和路径
            new_image_name = f"{start_index}_{original_image_name.split('_')[-1]}"
            new_image_path = target_folder / new_image_name

            # 复制图片
            shutil.copy(original_image_path, new_image_path)

        # 更新编号
        start_index += 1

print("图片复制完成。")