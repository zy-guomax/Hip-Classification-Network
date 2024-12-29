import os

def rename_images(folder_path):
    # 存储图片的四位数编号和对应的命名序号
    sequence_dict = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and "_" in filename:
            parts = filename.split("_")
            if len(parts) == 2 and len(parts[0]) == 4:
                prefix = parts[0]  # 四位数编号
                suffix = parts[1][0:4]  # 五位数编号

                # 如果四位数编号不在字典中，添加进去，并设置命名序号为1000
                if prefix not in sequence_dict:
                    sequence_dict[prefix] = 1000

                # 新的两位数编号
                new_suffix = "{:04d}_{:02d}".format(sequence_dict[prefix], int(suffix))

                # 更新图片的命名序号
                sequence_dict[prefix] += 1

                # 构造原文件的完整路径和新文件的完整路径
                original_filepath = os.path.join(folder_path, filename)
                new_filename = "{}_{}.png".format(prefix, new_suffix)
                new_filepath = os.path.join(folder_path, new_filename)

                # 重命名文件
                os.rename(original_filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'")

# 定义给定文件夹路径
folder_path = 'C:/Users/guozy/Desktop/data/non_collapse'  # 替换为你的文件夹路径

# 调用函数
rename_images(folder_path)