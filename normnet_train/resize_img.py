from PIL import Image
import os

def process_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 打开图片
            with Image.open(file_path) as img:
                # 首先将图片缩放到192x192
                img = img.resize((192, 192), Image.ANTIALIAS)
                # 然后裁剪中心区域到192x160
                width, height = img.size
                new_width = 160
                new_height = 192
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2

                img = img.crop((int(left), int(top), int(right), int(bottom)))
                # 以原文件名覆盖保存图片
                img.save(file_path)
                print(f'Processed and saved: {filename}')

# 调用函数，替换'path_to_your_folder'为你的图片文件夹路径
process_images('C:/Users/guozy/Desktop/data/coronal')
