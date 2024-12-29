import os
import time
import torch
import random
import logging
import numpy as np
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
from torch import nn
from scipy.stats import norm


def determine_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def update_config(config, cfgfile):
    config.defrost()
    config.merge_from_file(cfgfile)
    config.freeze()


def create_logger(out_dir, logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = os.path.join(out_dir, logfile[:-4]+'@'+time.strftime('(%m-%d)-%H-%M-%S')+'.log')

    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, times=1):
        self.val = val
        self.sum += val * times
        self.count += times
        self.avg = self.sum / self.count


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.7):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_img, neg_img):
        # # 计算InfoNCE损失
        record = []
        for img in pos_img:
            record.append(torch.norm(img, p=2).item())
        for img in neg_img:
            record.append(torch.norm(img, p=2).item())
        mu, std = norm.fit(record)
        pos_sim = 0
        for img in pos_img:
            pos_score = (torch.norm(img, p=2) - mu) / (std * self.temperature)
            pos_sim += torch.exp(pos_score)
            # print('pos_tensor', torch.norm(img, p=2).item())
            # print('pos_sim', pos_score.item())
        neg_sim = 0
        for img in neg_img:
            neg_score = (torch.norm(img, p=2) - mu) / (std * self.temperature)
            neg_sim += torch.exp(neg_score)
        loss = -torch.log(pos_sim/(pos_sim+neg_sim))
        # 计算InfoNCE损失
        return loss
