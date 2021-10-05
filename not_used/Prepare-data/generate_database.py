from PIL import Image
import numpy as np
import random
import os
import pandas
import json
import csv
import argparse
import torch
from pytorch_ssim import SSIM
import matplotlib.pyplot as plt
ssim_loss = SSIM()

def parse_config():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--image_num", type=int, default=500)
    parser.add_argument("--database_root", type=str, 
                                            default="/home/x227guo/workspace/SYDE675/Pixel_to_pixel/data", 
                                            help="original database's root path")
    parser.add_argument("--dataset_root", type=str, 
                                            default="/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset", 
                                            help="new dataset's root path")

    return parser.parse_args()

def Mkdir(new_dir):
    if not os.path.exists(new_dir):
        # to make intermediate folders
        os.makedirs(new_dir)
        #os.mkdir(new_dir)

def generate_database(config):

    image_num = config.image_num * 4 * 5 # 4 distortion type and 5 levels
    database_root = config.database_root
    new_dataset_root = config.dataset_root

    database_txt_name = os.path.join(database_root, "Waterloo_Exploration/Waterloo_Exploration.txt")

    list_of_database = pandas.read_csv(database_txt_name, sep=',', header=None)
    all_ref_image_path = []
    all_dis_image_path = []
    print('image number: ', list_of_database.shape[0]-1)
    for i in range(1, list_of_database.shape[0]):
        
        # seems read the wrong column
        all_ref_image_path.append(os.path.join(database_root, list_of_database.iloc[i, 2]) )
        all_dis_image_path.append(os.path.join(database_root, list_of_database.iloc[i, 0]) )

    for i in range(image_num):

        # first get image id
        ref_img_filename = all_ref_image_path[i]
        dis_img_filename = all_dis_image_path[i]
        image_id = ref_img_filename.split('.')[0].split('/')[-1]

        # Then create folders
        image_folder = os.path.join(new_dataset_root, image_id)
        Mkdir(image_folder)
        dis_folder = os.path.join(image_folder, 'dis')
        map_folder = os.path.join(image_folder, 'map')
        Mkdir(dis_folder)
        Mkdir(map_folder)
        j = 0 # patch format from previous codes
        dis_patch_fullname = os.path.join(dis_folder, str(j).zfill(5) + '.png')
        map_patch_fullname = os.path.join(map_folder, str(j).zfill(5) + '.npz')

        # calculate ssim map
        ref_img_np_or = np.array(Image.open(ref_img_filename).convert("L"), dtype=np.float32)
        dis_img_np_or = np.array(Image.open(dis_img_filename).convert("L"), dtype=np.float32)
        ref_img_np = np.expand_dims(ref_img_np_or, axis=0)
        ref_img_np = np.expand_dims(ref_img_np, axis=0)
        dis_img_np = np.expand_dims(dis_img_np_or, axis=0)
        dis_img_np = np.expand_dims(dis_img_np, axis=0)
        ref_img_np /= 255
        dis_img_np /= 255
        ref_img = torch.tensor(ref_img_np)
        dis_img = torch.tensor(dis_img_np)
        ref_img = ref_img.cuda()
        dis_img = dis_img.cuda()
        # print(ref_img_np.shape)

        ssim_map = ssim_loss(ref_img, dis_img).cpu().detach().numpy()
        ssim_map = np.squeeze(ssim_map)
        # plt.imshow(ssim_map, cmap='gray')
        # plt.show()
        # print(ssim_map.mean())
        # 1/0

        dis_patch_img = Image.fromarray(dis_img_np_or.astype(np.uint8)) 

        # save it!
        plt.imsave(dis_patch_fullname, dis_img_np_or, cmap='gray')
        np.savez(map_patch_fullname, qmap=ssim_map)

        if i % 100 == 0:
            print(i)


if __name__ == "__main__":
    cfg = parse_config()
    generate_database(cfg)

