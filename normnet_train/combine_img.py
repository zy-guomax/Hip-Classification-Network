import os
import shutil


def merge_coronal_images(source_folder, target_folder):
    # 创建目标文件夹，如果它不存在的话
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹
    for root, dirs, files in os.walk(source_folder):
        # 寻找名为 'coronal' 的子文件夹
        if 'Coronal' in dirs:
            coronal_path = os.path.join(root, 'coronal')
            subdir_name = os.path.basename(root)
            # 遍历 'coronal' 文件夹中的所有文件
            for file in os.listdir(coronal_path):
                # 检查文件扩展名是否为 '.png'
                if file.lower().endswith('.png'):
                    # 构建完整的文件路径
                    file_path = os.path.join(coronal_path, file)
                    # 构建目标文件路径
                    target_file_name = f"{subdir_name}_{file}"
                    target_file_path = os.path.join(target_folder, target_file_name)
                    # 复制文件
                    shutil.copy(file_path, target_file_path)
                    print(f"Copied: {file_path} to {target_file_path}")


# 调用函数
source_folder = "C:/Users/guozy/Desktop/data/dataset_final_normal"  # 替换为你的源文件夹路径
target_folder = "C:/Users/guozy/Desktop/data/normal"  # 替换为你的目标文件夹路径，这将是新创建的文件夹
merge_coronal_images(source_folder, target_folder)
