import os
import sys
import torch
import logging
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from .base_data_loader import BaseDataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image 

class Dataset(data.Dataset):
    def __init__(self, data_dir):

        origin_dir = data_dir + 'mixed/'
        img_list = os.listdir(origin_dir)
        self.data_dir = data_dir
        self.img_list = img_list
        self.transform=transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, index):
        origin_img = self.pil_loader(self.data_dir + 'mixed/' + self.img_list[index])
        hw_img = self.pil_loader(self.data_dir + 'handwritten/' + self.img_list[index])
        pt_img = self.pil_loader(self.data_dir + 'form/' + self.img_list[index])
        return origin_img, hw_img, pt_img

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = img.resize((170, 220), Image.ANTIALIAS)
            img = np.array(img).astype(np.float32)
            img = 255.0 - img
            img = (img - 127.5) / 127.5
            img = np.expand_dims(img, axis = 2)
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)

class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers = 0, training=True):
        dataset = Dataset(data_dir)
        collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)

