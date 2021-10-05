import torch
from fcn import U_net
import functools
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn

"""

generate demo-distorted image & ssim quality map pair
perform testing on a single image

"""

path_to_project = "/home/x227guo/workspace/SYDE675/Pixel_to_pixel"

unet_ckpt_path = path_to_project + "/checkpoint/test_u-net_fixdatasetbug/U_net-00979.pt"
cgan_ckpt_path = path_to_project + "/checkpoint/test_conditional_gan_unet256/latest_net_G.pth"
all_ckpt_path = [unet_ckpt_path, cgan_ckpt_path]


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

# unet 256
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
            (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load_networks(net, load_path):
        
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(load_path)

    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))

    net.load_state_dict(state_dict)

    return net


def produce_ssimmap(model, image):

    image = image.cuda()
    ssimmap = model(image)
    print(ssimmap.shape)


def get_crop(img, qmap):

    # random crop
    # this can solve random issue, avoid cropping at different locations
    # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop, self.crop))
    i, j, h, w = 0, 0, 256, 256
    img = transforms.functional.crop(img, i, j, h, w)
    qmap = qmap[i:i+h, j:j+w]

    return img, qmap


def load_image(transform):

    image_path = "/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset/valid/00003_1_1/dis/00000.png"
    ssimmap_path = "/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset/valid/00003_1_1/map/00000.npz"
    dis_img = Image.open(image_path)
    qmap_np = np.load(ssimmap_path)['qmap']
    dis_img, qmap_np = get_crop(dis_img, qmap_np)

    dis_img = transform(dis_img)
    
    return dis_img, qmap_np


def main():

    transform = transforms.Compose([transforms.ToTensor()])
    for ckpt_path in all_ckpt_path:

        print(ckpt_path)
        if "u-net" in ckpt_path:
            print("loading unet")
            model = U_net()
            model = nn.DataParallel(model)
            model.cuda()
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['state_dict'])
        elif "conditional_gan" in ckpt_path:
            model = UnetGenerator(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False))
            model = nn.DataParallel(model)
            model.cuda()
            model = load_networks(model, ckpt_path)
        else:
            raise NotImplementedError(f"[****] '{ckpt_path}' is not a valid path")
    
    
        dis_img, qmap_np = load_image(transform)
        dis_img = torch.unsqueeze(dis_img, 0)
        #print(dis_img.shape, qmap_np.shape)

        produce_ssimmap(model, dis_img)


if __name__ == "__main__":
   
   main()


