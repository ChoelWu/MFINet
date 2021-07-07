import math
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class TrainDataset(Dataset):
    def __init__(self, imgs, masks, data_augmentation=False):
        self.imgs = imgs
        self.masks = masks
        self.data_augmentation = data_augmentation
        self.p = 0.5

    def __len__(self):
        return self.imgs.shape[0]

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __getitem__(self, idx):
        img = np.reshape(self.imgs[idx], (self.imgs[idx].shape[1], self.imgs[idx].shape[2]))
        mask = np.reshape(self.masks[idx], (self.masks[idx].shape[1], self.masks[idx].shape[2]))
        h, w = img.shape[0], img.shape[1]
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        if self.data_augmentation is True and random.random() < 0.6:
            # 随机水平翻转
            if random.random() > self.p:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机垂直翻转
            if random.random() > self.p:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # 逆时针90度
            if random.random() > self.p:
                img = img.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)
            # 逆时针180度
            if random.random() > self.p:
                img = img.transpose(Image.ROTATE_180)
                mask = mask.transpose(Image.ROTATE_180)
            # 逆时针270度
            if random.random() > self.p:
                img = img.transpose(Image.ROTATE_270)
                mask = mask.transpose(Image.ROTATE_270)
            # 随机裁剪
            img = TF.pad(img, int(math.ceil(h * 1 / 8)))
            mask = TF.pad(mask, int(math.ceil(h * 1 / 8)))
            i, j, h, w = self.get_params(img, (h, w))
            img = img.crop((j, i, j + w, i + h))
            mask = mask.crop((j, i, j + w, i + h))

        return TF.to_tensor(img), torch.from_numpy(np.array(mask)).long()


class TestDataset(Dataset):
    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx, ...]).float()
