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
from PIL import Image
ssim_loss = SSIM()

import time
from multiprocessing import Pool


def parse_config():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--image_num", type=int, default=500)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--dataset_root", type=str, 
                                            default="/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset_for_ppt_demo/dataset", 
                                            help="new dataset's root path")
    parser.add_argument("--p2p_dataset_root", type=str, 
                                            default="/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset_for_ppt_demo", 
                                            help="new dataset's root path")
                            

    return parser.parse_args()


def Mkdir(new_dir):
    if not os.path.exists(new_dir):
        #os.mkdir(new_dir)
        os.makedirs(new_dir)


def read_and_save(config, dis_patch_name, qmap_patch_name, save_name):
    
    dis_img = Image.open(dis_patch_name)
    qmap_np = np.load(qmap_patch_name)['qmap']
    qmap_np = Image.fromarray((qmap_np*128+127).astype(np.uint8), 'L')
    
    # save
    new_dis_patch_dir = config.p2p_dataset_root + "/A/test/"
    new_map_patch_dir = config.p2p_dataset_root + "/B/test/"

    Mkdir(new_dis_patch_dir)
    Mkdir(new_map_patch_dir)

    dis_patch_fullname = new_dis_patch_dir + save_name
    map_patch_fullname = new_map_patch_dir + save_name

    dis_img.save(dis_patch_fullname)
    qmap_np.save(map_patch_fullname)


def change_npz_to_png(config):

    old_dataset_path = config.dataset_root
    new_dataset_path = config.p2p_dataset_root


    # For generating the demo
    for i, sub_dir in enumerate(os.listdir(old_dataset_path)):

        save_name = str(sub_dir) + ".png"

        # read
        sub_dir = old_dataset_path + "/" + sub_dir
        image_dir = sub_dir + "/dis"
        map_dir = sub_dir + "/map"
        dis_patch_name = image_dir + "/00000.png"
        qmap_patch_name = map_dir + "/00000.npz"

        # time used: 1.4s
        #pool.apply_async(read_and_save, args=(config, dis_patch_name, qmap_patch_name, save_name))

        # time used: 13s
        read_and_save(config, dis_patch_name, qmap_patch_name, save_name)
        



    # # For train data
    # for i, sub_dir in enumerate(os.listdir(old_dataset_path + "/train")):

    #     save_name = str(i) + ".png"
        
    #     # read
    #     sub_dir = old_dataset_path + "/train/" + sub_dir
    #     image_dir = sub_dir + "/dis"
    #     map_dir = sub_dir + "/map"
    #     dis_patch_name = image_dir + "/00000.png"
    #     qmap_patch_name = map_dir + "/00000.npz"

    #     dis_img = Image.open(dis_patch_name)
    #     qmap_np = np.load(qmap_patch_name)['qmap']
    #     qmap_np = Image.fromarray((qmap_np*128+127).astype(np.uint8), 'L')
    #     #qmap_np = Image.fromarray(qmap_np, 'L')

    #     # save
    #     new_dis_patch_dir = new_dataset_path + "/A/train/"
    #     new_map_patch_dir = new_dataset_path + "/B/train/"
    #     Mkdir(new_dis_patch_dir)
    #     Mkdir(new_map_patch_dir)

    #     dis_patch_fullname = new_dis_patch_dir + save_name
    #     map_patch_fullname = new_map_patch_dir + save_name
    #     dis_img.save(dis_patch_fullname)
    #     qmap_np.save(map_patch_fullname)
    
    #  # For valid data
    # for i, sub_dir in enumerate(os.listdir(old_dataset_path + "/valid")):

    #     save_name = str(i) + ".png"
        
    #     # read
    #     sub_dir = old_dataset_path + "/valid/" + sub_dir
    #     image_dir = sub_dir + "/dis"
    #     map_dir = sub_dir + "/map"
    #     dis_patch_name = image_dir + "/00000.png"
    #     qmap_patch_name = map_dir + "/00000.npz"
    
    #     dis_img = Image.open(dis_patch_name)
    #     qmap_np = np.load(qmap_patch_name)['qmap']

    #     # +128 will turn white to black
    #     # no converting to uint8 make output ssimmap strange
    #     qmap_np = Image.fromarray((qmap_np*128+127).astype(np.uint8), 'L') 
    #     #qmap_np = Image.fromarray((qmap_np*128+128).astype(np.uint8), 'L')    # result in entire black image
    #     #qmap_np = Image.fromarray(qmap_np, 'L')

    #     # save
    #     new_dis_patch_dir = new_dataset_path + "/A/valid/"
    #     new_map_patch_dir = new_dataset_path + "/B/valid/"
    #     Mkdir(new_dis_patch_dir)
    #     Mkdir(new_map_patch_dir)

    #     dis_patch_fullname = new_dis_patch_dir + save_name
    #     map_patch_fullname = new_map_patch_dir + save_name

    #     dis_img.save(dis_patch_fullname)
    #     qmap_np.save(map_patch_fullname)


if __name__ == "__main__":
    cfg = parse_config()

    start1 = time.time()
    pool=Pool()

    change_npz_to_png(cfg)

    pool.close()
    pool.join()
    print("time used:", time.time()-start1)

    # start1 = time.time()
    
    # print("time used:", time.time()-start1)
    

