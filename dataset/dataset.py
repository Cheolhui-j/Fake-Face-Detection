import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os, random

def default_loader(path):
    #size = random.randint(64, 256)

    # im = cv2.imread(path)
    # im = cv2.resize(im, (size, size))
    # im = cv2.resize(im, (64, 64))
    # ims = np.zeros((3, 64, 64))
    # ims[0, :, :] = im[:, :, 0]
    # ims[1, :, :] = im[:, :, 1]
    # ims[2, :, :] = im[:, :, 2]
    # img_tensor = torch.tensor(ims.astype('float32'))
    img = Image.open(path)
    img = img.resize((64, 64))
    return img


class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[-1], line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label