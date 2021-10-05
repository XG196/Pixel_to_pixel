## Pixel_to_pixel

### Package download
#### conda install -c pytorch torchvision cudatoolkit=11.0 pytorch
#### conda install -c conda-forge tensorboard


### Instructions for useing Pix2Pix model to do image-to-image translation (cGAN)

### 1. Dateset
#### (1). The image should be place somewhere like this path_to_images/A/train, path_to_images/B/train. A and B are for example, edge images and color images.
#### (2). Use combine_A_and_B.py under datasets directory to combine A and B to one image placed under path_to_images/AB/train

### 2. Training
#### python train.py --dataroot /home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset_for_p2p/AB --name ssimmap_pix2pix --model pix2pix --netG unet_128 --direction AtoB --batch_size 50
#### name is the experiment name, model is pix2pix model and the net structure for generator can be chosen to be Resnet or Unet structure. The direction AtoB means
#### learning map from A to B, and finally the batchsize. The netG will take data shape (50,3, crop_size, crop_size) as input.
#### More optitions can see base_options.py and train_options.py under options folder.

### 3. visulization
#### During training the fake_B, real_B real_A and the loss can be visulized. You should first install visdom and use command "python -m visdom.server". 
#### The display port is specified in the train_options.py file.
