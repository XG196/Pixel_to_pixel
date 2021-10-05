import json
import os
import random
import pandas
import torch

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class qmapDataset(torch.utils.data.Dataset):
    # maybe change to download from Internet later ...
    
    def __init__(self, dataroot, transforms, patch_num=50, crop_size=224):
        self.root = dataroot
        self.img_list = os.listdir(self.root)

        self.dis_size = len(self.img_list)
        self.patch_num = patch_num

        self.transforms = transforms
        self.crop = crop_size

    def __len__(self):
        return self.dis_size

    def get_crop(self, img, qmap):
        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop, self.crop))
        img = transforms.functional.crop(img, i, j, h, w)
        qmap = qmap[i:i+h, j:j+w]

        return img, qmap

    def __getitem__(self, idx):

        dis_img_name = self.img_list[idx]
        patch_folder = os.path.join(self.root, dis_img_name)
        
        dis_folder = os.path.join(patch_folder, 'dis')
        qmap_folder = os.path.join(patch_folder, 'map')

        read = False
        while not read:

            patch_selection = random.randint(0, self.patch_num-1)
            patch_name = str(patch_selection).zfill(5) + '.png'
            qmap_name = str(patch_selection).zfill(5) + '.npz'

            dis_patch_name = os.path.join(dis_folder, patch_name)
            qmap_patch_name = os.path.join(qmap_folder, qmap_name)

            # read qmap
            try:
                qmap_np = np.load(qmap_patch_name)['qmap']
                dis_img = Image.open(dis_patch_name)
                dis_img, qmap_np = self.get_crop(dis_img, qmap_np)
                dis_img = self.transforms(dis_img)

                qmap_np = np.expand_dims(qmap_np, axis=0)
                qmap_np = qmap_np.astype(np.float32)

                sum_qmap = np.sum(qmap_np)
                read = True
            except:
                print(qmap_patch_name)
                1/0
                continue
            
            if np.isnan(sum_qmap):
                print(qmap_patch_name)
                1/0

        # print(dis_img.shape)
        # print(qmap_np.shape)

        sample = {'dis': dis_img, 'qmap': qmap_np}
        return sample
