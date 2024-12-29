import os

def rename_images(folder_path):
    # 存储图片的四位数编号和对应的命名序号
    sequence_dict = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and "_" in filename:
            parts = filename.split("_")
            if len(parts) == 2 and len(parts[0]) == 4: # and len(parts[1]) == 2:
                # 检查四位数是否以2开头
                if parts[0].startswith('1'):
                    # 构造新的四位数，以1开头
                    new_prefix = '1' + parts[0][1:]
                    new_filename = f"{new_prefix}_{parts[1][:-4]}"
                    # 构造原文件的完整路径和新文件的完整路径
                    original_filepath = os.path.join(folder_path, filename)
                    new_filepath = os.path.join(folder_path, new_filename)
                    # 重命名文件
                    os.rename(original_filepath, new_filepath)
                    print(f"Renamed '{filename}' to '{new_filename}'")

# 定义给定文件夹路径
folder_path = 'C:/Users/guozy/Desktop/data/non_collapse'  # 替换为你的文件夹路径

# 调用函数
rename_images(folder_path)