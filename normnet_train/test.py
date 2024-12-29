import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import numpy as np
import os
from PIL import Image
from normnet import Model
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F


class TestDataset(Dataset):
    # 导入所有数据，不含标签，不作预处理/增强
    def __init__(self, root_dir, transform_img):
        self.base_folder = root_dir
        # self.data = datasets.ImageFolder(root_dir, transform=transform_img)
        self.transform = transform_img

    # 访问数据集大小
    def __len__(self):
        # return len(self.data)
        return 13804

    def __getitem__(self, index):
        filename = []
        if os.path.exists(self.base_folder) and os.path.isdir(self.base_folder):
            for file_name in os.listdir(self.base_folder):
                filename.append(file_name)
            ind = random.randint(11043, 13803)
            file = filename[ind]
            image_path = os.path.join(self.base_folder, file)
            img = Image.open(image_path)
            img = self.transform(img)
        return img


# 单独二维图像
model = Model()
model.load_state_dict(torch.load('/root/model.pt'))
model.eval()  # 设置为评估模式

# 关闭梯度更新
for param in model.parameters():
    param.requires_grad = False

# Hyperparameters
batch_size = 32

# 创建保存输出的路径
output_dir = '/root/test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((192, 160)),
    transforms.RandomHorizontalFlip(),                # 随机水平旋转
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])


# Create data loaders
test_dataset = TestDataset(root_dir='/root/autodl-fs/coronal', transform_img=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, data in enumerate(test_loader, 0):  # data 为一个batch_size（32）的张量
    with torch.no_grad():  # 禁用梯度计算
        output_e0, output_d3 = model(data)
        # 将输出转换为图片并保存
        e0_img = transforms.ToPILImage()(output_e0.squeeze().cpu())
        d3_img = transforms.ToPILImage()(output_d3.squeeze().cpu())
        e0_img.save(os.path.join(output_dir, f'output_e0_{i}.png'))
        d3_img.save(os.path.join(output_dir, f'output_d3_{i}.png'))
