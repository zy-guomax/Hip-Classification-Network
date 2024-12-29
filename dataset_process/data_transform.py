import os
from PIL import Image
from contrast_net.dataset.data_augment import transform
from torchvision.transforms import functional as F


# 定义给定文件夹路径
folder_path = 'C:/Users/guozy/Desktop/data/coronal_normal'  # 替换为你的文件夹路径

# 确保文件夹路径存在
if not os.path.exists(folder_path):
    print(f"路径 {folder_path} 不存在。")
else:
    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and "_" in filename:
            # 读取图片文件
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.convert('L')

            # 获取文件名的各部分
            parts = filename.split("_")
            if len(parts) == 3 and len(parts[0]) == 4:
                prefix = parts[0]  # 四位数
                first_suffix = parts[1]  # 第一个两位数

                # 对图片进行15次变换
                for i in range(1, 16):
                    transformed_image = transform(image)  # 应用变换
                    transformed_image = F.to_pil_image(transformed_image)
                    transformed_image = transformed_image.convert('L')

                    new_filename = f"{prefix}_{first_suffix}_{str(i).zfill(2)}.png"  # 创建新的文件名
                    new_image_path = os.path.join(folder_path, new_filename)  # 创建新的文件路径

                    # 保存变换后的图片
                    transformed_image.save(new_image_path)
                    print(f"已保存变换后的图片：{new_image_path}")