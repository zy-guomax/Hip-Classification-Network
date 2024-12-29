import numpy as np
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image, ImageFilter, ImageEnhance
import random


# 定义高斯噪声变换
def gaussian_noise_transform(p_per_sample):
    def noise_transform(image):
        if random.random() < p_per_sample:
            mean = 0
            std = 0.1 * 255
            np.random.seed(int(np.random.random() * 1000))
            noise = np.random.normal(mean, std, (image.height, image.width))
            noisy_image = np.array(image) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_image)
        else:
            return image
    return noise_transform


# 定义高斯模糊变换
def gaussian_blur_transform(blur_sigma, different_sigma_per_channel, p_per_sample):
    def blur_transform(image):
        if random.random() < p_per_sample:
            if different_sigma_per_channel:
                sigma = random.uniform(*blur_sigma)
            else:
                sigma = blur_sigma[0]
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            return image
    return blur_transform


# 定义亮度调整变换
def brightness_transform(multiplier_range, p):
    def transform_brightness(image):
        if random.random() < p:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(*multiplier_range)
            return enhancer.enhance(factor)
        else:
            return image
    return transform_brightness


# 定义对比度调整变换
def contrast_transform(p):
    def transform_contrast(image):
        if random.random() < p:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.75, 1.25)  # 假设对比度因子范围与亮度相同
            return enhancer.enhance(factor)
        else:
            return image
    return transform_contrast


class SimulateLowResolutionTransform:
    def __init__(self, zoom_range, per_channel, p_per_channel, order_downsample, order_upsample, p_per_sample):
        self.zoom_range = zoom_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.order_downsample = order_downsample
        self.order_upsample = order_upsample
        self.p_per_sample = p_per_sample

    def __call__(self, image):
        if random.random() < self.p_per_sample:
            scale_factor = random.uniform(*self.zoom_range)
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)

            # 执行下采样，使用指定的插值方法
            image = image.resize((new_width, new_height), resample=self.order_downsample)

            # 执行上采样，使用指定的插值方法
            image = image.resize((image.width * self.order_upsample, image.height * self.order_upsample), resample=self.order_upsample)
        return image


# 定义伽马变换
def gamma_transform(gamma_range, invert_image, p_per_sample):
    def transform_gamma(image):
        if random.random() < p_per_sample:
            gamma = random.uniform(*gamma_range)
            if invert_image:
                gamma = 1 / gamma
            return functional.adjust_gamma(image, gamma, gain=1)
        else:
            return image
    return transform_gamma


# 定义各变换
angle_x = (-np.pi, np.pi)
gaussian_noise = gaussian_noise_transform(p_per_sample=0.1)
gaussian_blur = gaussian_blur_transform(blur_sigma=(0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2)
brightness = brightness_transform((0.75, 1.25), 0.15)
contrast = contrast_transform(0.15)
# 创建SimulateLowResolutionTransform的实例
low_resolution_transform = SimulateLowResolutionTransform(
    zoom_range=(0.5, 1.0),
    per_channel=True,
    p_per_channel=0.5,
    order_downsample=0,
    order_upsample=3,
    p_per_sample=0.25
)
gamma_transform1 = gamma_transform((0.7, 1.5), invert_image=True, p_per_sample=0.1)
gamma_transform2 = gamma_transform((0.7, 1.5), invert_image=False, p_per_sample=0.3)

# 使用RandomApply以50%的概率应用每个变换
transform = transforms.Compose([
    transforms.Resize((192, 160)),  # 调整大小
    transforms.RandomApply([gaussian_noise], p=0.5),
    transforms.RandomApply([gaussian_blur], p=0.5),
    transforms.RandomApply([brightness], p=0.5),
    transforms.RandomApply([contrast], p=0.5),
    transforms.RandomApply([low_resolution_transform], p=0.5),
    transforms.RandomApply([gamma_transform1], p=0.5),
    transforms.RandomApply([gamma_transform2], p=0.5),
])