from core.config import config
import numpy as np
import random
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class HipDataset(Dataset):
    def __init__(self, file_path, batch_size):
        super().__init__()
        self.txt_file_path = file_path
        self.root = '/root/autodl-fs'
        self.file_names = self._load_file_names()
        self.transform = transforms.Compose([
            transforms.Resize((192,160)),
            transforms.ToTensor()
        ])
        self.thread_id = None  # 添加线程ID属性
        self.batch_size = batch_size

    # 访问数据集大小
    def __len__(self):
        return 7140

    def _load_file_names(self):
        # 读取txt文件中的所有文件名
        with open(self.txt_file_path, 'r') as file:
            file_names = [line.strip() for line in file.readlines()]
        # 过滤出格式正确的文件名
        valid_file_names = [fn for fn in file_names if len(fn.split('_')) == 2]
        return valid_file_names

    def _select_file_name(self):  # 深度
        # 依据正态概率选择一个二位数
        mean = 20  # 均值为20，因为范围是00-40
        std = 10  # 标准差为10
        selected_digit = np.random.normal(mean, std)
        selected_digit = int(round(selected_digit))

        # 确保选择的数字在00-40范围内
        if selected_digit < 0 or selected_digit > 50:
            return self._select_file_name()

        # 在所有第一个二位数为该二位数的文件名中随机选择一个
        norm_names = [fn for fn in self.file_names if int(fn.split('_')[1][0:2]) == selected_digit and fn.split('_')[0].startswith('1')]  # normal file
        train_names = [fn for fn in self.file_names if int(fn.split('_')[1][0:2]) == selected_digit]
        return random.choice(norm_names), random.choice(train_names)

    def load_image_data(self, file_name):
        if file_name.split('_')[0].startswith('1'):
            tar_folder = 'non_collapse'
            label = 0
        else:
            tar_folder = 'collapse'
            label = 1
        tar_folder_path = os.path.join(self.root, tar_folder)
        image_path = os.path.join(tar_folder_path, file_name)
        img = self.transform(Image.open(image_path).convert('L'))
        return img, float(label)

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id  # 设置线程ID

    def __getitem__(self, index):
        norm_name, train_name = self._select_file_name()
        # 读取图片数据（这里假设有一个函数可以加载图片数据）
        norm_img, label = self.load_image_data(norm_name)
        train_img, label = self.load_image_data(train_name)

        return [train_img, norm_img], label


def get_train_loader():
    train_txt_path = '/root/autodl-fs/mlp_train.txt'
    dataset = HipDataset(train_txt_path, config.TRAIN.BATCH_SIZE)
    dataloader = DataLoader(dataset, 4, shuffle=True)
    return dataloader


def get_valid_loader():
    valid_txt_path = '/root/autodl-fs/mlp_valid.txt'
    dataset = HipDataset(valid_txt_path, config.TRAIN.BATCH_SIZE)
    dataloader = DataLoader(dataset, 4, shuffle=True)
    return dataloader
