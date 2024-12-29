import os
import re

# 定义给定文件夹路径
folder_path = 'C:/Users/guozy/Desktop/data/non_collapse'  # 替换为你的文件夹路径

# 确保文件夹路径存在
if not os.path.exists(folder_path):
    print(f"路径 {folder_path} 不存在。")
else:
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 使用正则表达式匹配文件名格式
        match = re.match(r'(\d{4})_(\d{5})\.png', filename)
        if match:
            # 提取四位数和五位数后缀
            prefix = match.group(1)
            suffix = int(match.group(2))

            # 检查五位数后缀是否在00020-00060之间
            if not 15 <= suffix <= 65:
                # 构造完整的文件路径
                file_path = os.path.join(folder_path, filename)

                # 删除文件
                os.remove(file_path)
                print(f"已删除文件：{filename}")

print("删除操作完成。")