from core.config import config
import numpy as np
import random
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.data_augment import transform


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
        valid_file_names = [fn for fn in file_names if len(fn.split('_')) == 3]
        return valid_file_names

    def _select_file_name(self):  # 深度
        # 依据正态概率选择一个二位数
        mean = 20  # 均值为20，因为范围是00-40
        std = 10  # 标准差为10
        selected_digit = np.random.normal(mean, std)
        selected_digit = int(round(selected_digit))

        # 确保选择的数字在00-40范围内
        if selected_digit < 0 or selected_digit > 40:
            return self._select_file_name()

        # 在所有第一个二位数为该二位数的文件名中随机选择一个
        filtered_names = [fn for fn in self.file_names if int(fn.split('_')[1]) == selected_digit and fn.split('_')[0].startswith('1')]  # normal file
        return random.choice(filtered_names)

    def load_image_data_norm(self, file_name):
        tar_folder = 'coronal_normal'
        tar_folder_path = os.path.join(self.root, tar_folder)
        norm_image_path = os.path.join(tar_folder_path, file_name)
        norm_img = Image.open(norm_image_path)
        return norm_img

    def load_image_data_pos(self, file_name):
        tar_folder = 'coronal_normal'
        tar_folder_path = os.path.join(self.root, tar_folder)
        norm_image_path = os.path.join(tar_folder_path, file_name)
        norm_img = Image.open(norm_image_path)
        pos_img = transform(norm_img)
        return pos_img

    def load_image_data_neg(self, file_name):
        depth = int(file_name.split('_')[1])
        random_number = random.randint(max(0, depth-5), min(40, depth+5))
        file_name = random.choice([fn for fn in self.file_names if int(fn.split('_')[1]) == random_number and fn.split('_')[0].startswith('2')])  # normal file
        tar_folder = 'coronal_patient'
        tar_folder_path = os.path.join(self.root, tar_folder)
        neg_image_path = os.path.join(tar_folder_path, file_name)
        neg_img = Image.open(neg_image_path)
        return neg_img

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id  # 设置线程ID

    def __getitem__(self, index):
        file_name = self._select_file_name()
        # 读取图片数据（这里假设有一个函数可以加载图片数据）
        norm_img = self.load_image_data_pos(file_name)
        norm_img = self.transform(norm_img)
        pos_img = []
        for i in range(self.batch_size):
            img = self.transform(self.load_image_data_pos(file_name))
            pos_img.append(img)
        neg_img = []
        for i in range(self.batch_size):
            img = self.transform(self.load_image_data_neg(file_name))
            neg_img.append(img)

        return norm_img, pos_img, neg_img


def get_train_loader():
    train_txt_path = '/root/autodl-fs/train.txt'
    dataset = HipDataset(train_txt_path, config.TRAIN.BATCH_SIZE)
    dataloader = DataLoader(dataset, 1, shuffle=True)
    return dataloader


def get_valid_loader():
    valid_txt_path = '/root/autodl-fs/valid.txt'
    dataset = HipDataset(valid_txt_path, config.TRAIN.BATCH_SIZE)
    dataloader = DataLoader(dataset, 1, shuffle=True)
    return dataloader
