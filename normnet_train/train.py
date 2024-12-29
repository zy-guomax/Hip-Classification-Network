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


# 打乱axial顺序
folder_path = '/root/autodl-fs/normal'
# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)
# 打乱文件顺序
random.shuffle(file_names)


class CustomDataset(Dataset):
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
            ind = random.randint(0, 8281)
            file = filename[ind]
            image_path = os.path.join(self.base_folder, file)
            img = Image.open(image_path)
            img = self.transform(img)
        return img


class ValidDataset(Dataset):
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
            ind = random.randint(8282, 11042)
            file = filename[ind]
            image_path = os.path.join(self.base_folder, file)
            img = Image.open(image_path)
            img = self.transform(img)
        return img


def plot_losses(loss_values):
    iterations = range(1, len(loss_values) + 1)  # 迭代次数
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_values, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.grid(True)
    plt.savefig('/root/loss.jpg')
    plt.close()


def plot_test_losses(loss_values):
    test_sum_iterations = range(1, len(loss_values) + 1)  # 迭代次数
    plt.figure(figsize=(8, 6))
    plt.plot(test_sum_iterations, loss_values, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Test_Loss')
    plt.title('Test_Loss vs Iteration')
    plt.grid(True)
    plt.savefig('/root/test_loss.jpg')
    plt.close()


# 单独二维图像
model = Model()

# Hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 100000  # 应增大至约100
loss_array = []
image_test_loss_array = []

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((192, 160)),
    transforms.RandomHorizontalFlip(),                # 随机水平旋转
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])


# Create data loaders
train_dataset = CustomDataset(root_dir='/root/autodl-fs/coronal', transform_img=transform)
Valid_dataset = ValidDataset(root_dir='/root/autodl-fs/coronal', transform_img=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
Valid_loader = DataLoader(Valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize the contrastive learning network and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
iteration = 0
min_loss = 1

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):  # data 为一个batch_size（32）的张量
        sample = data
        optimizer.zero_grad()
        output_e0, output_d3 = model(sample)
        loss = F.mse_loss(output_d3, output_e0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_array.append(loss.item())
        plot_losses(loss_array)
        iteration = iteration + 1
        if (iteration % 10) == 0:
            cur_loss = 2
            check_iteration = 0
            while cur_loss > min_loss:
                test_loss_array = []
                data_check = next(iter(Valid_loader))
                sample_check = data_check
                output_check_e0, output_check_d3 = model(sample_check)  # 提取特征向量
                cur_loss = F.mse_loss(output_check_d3, output_check_e0)  # 交叉熵计算loss
                test_loss_array.append(cur_loss.item())  # 数组存储损失值
                cur_loss = np.mean(test_loss_array)
                image_test_loss_array.append(cur_loss)
                plot_test_losses(image_test_loss_array)  # 画图
                if cur_loss < min_loss:
                    break
                check_iteration = check_iteration + 1
                if check_iteration > 5:
                    print('stop!')
                    sys.exit()
            min_loss = cur_loss
            torch.save(model.encoder.state_dict(), '/root/encoder.pt')
            torch.save(model.state_dict(), '/root/model.pt')
