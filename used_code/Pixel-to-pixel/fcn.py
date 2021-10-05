# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

"""

containing a set of models for dense prediction (map from image to image)

"""

class FCN_VGG(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        score = self.relu(self.deconv1(x5))     # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv3(score))  # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.deconv4(score))  # size=(N, 64, x.H/2, x.W/2)
        score = self.relu(self.deconv5(score))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        score = self.sigmoid(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCNs(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.maxpooling =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu    = nn.ReLU(inplace=True)
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.maxpooling(self.relu(self.conv1(x)))  # size = (N, 32, W/2, H/2)
        x2 = self.maxpooling(self.relu(self.conv2(x1))) # size = (N, 64, W/4, H/4)
        x3 = self.maxpooling(self.relu(self.conv3(x2))) # size = (N, 128, W/8, H/8)
        x4 = self.maxpooling(self.relu(self.conv4(x3))) # size = (N, 256, W/16, H/16)
        x5 = self.maxpooling(self.relu(self.conv5(x4))) # size = (N, 512, W/32, H/32)

        score = self.relu(self.deconv1(x5))     # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv3(score))  # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.deconv4(score))  # size=(N, 64, x.H/2, x.W/2)
        score = self.relu(self.deconv5(score))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        score = self.sigmoid(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class fcn_resnet50_p2p(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet101 = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet50', pretrained=pretrained)
        self.classifier = nn.Conv2d(21, 1, kernel_size=1)

    def forward(self, x):

        output = self.resnet101(x)
        x = output['out']  # size=(N, 21, x.H, x.W)

        score = self.classifier(x)                    # size=(N, 1, x.H, x.W)

        return score

class fcn_resnet101_p2p(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet101 = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=pretrained)
        self.classifier = nn.Conv2d(21, 1, kernel_size=1)

    def forward(self, x):

        output = self.resnet101(x)
        x = output['out']  # size=(N, 21, x.H, x.W)

        score = self.classifier(x)                    # size=(N, 1, x.H, x.W)

        return score

class fcn_densenet_p2p(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet101 = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=pretrained)
        self.classifier = nn.Conv2d(21, 1, kernel_size=1)

    def forward(self, x):

        output = self.resnet101(x)
        x = output['out']  # size=(N, 21, x.H, x.W)

        score = self.classifier(x)                    # size=(N, 1, x.H, x.W)

        return score

class U_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpooling =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu    = nn.LeakyReLU(inplace=True)
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.classifier = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.maxpooling(self.relu(self.conv1(x)))  # size = (N, 32, W/2, H/2)
        x2 = self.maxpooling(self.relu(self.conv2(x1))) # size = (N, 64, W/4, H/4)
        x3 = self.maxpooling(self.relu(self.conv3(x2))) # size = (N, 128, W/8, H/8)
        x4 = self.maxpooling(self.relu(self.conv4(x3))) # size = (N, 256, W/16, H/16)

        x = self.relu(self.deconv1(x4))                 # size = (N, 128, W/8, H/8)
        x = torch.cat((x3, x), 1)                                 # size = (N, 256, W/8, H/8)

        x = self.relu(self.deconv2(x))                  # size = (N, 64, W/4, H/4)
        x = torch.cat((x2, x), 1)                                 # size = (N, 128, W/4, H/4)

        x = self.relu(self.deconv3(x))                  # size = (N, 32, W/2, H/2)
        x = torch.cat((x1, x), 1)                                 # size = (N, 64, W/2, H/2)

        x = self.relu(self.deconv4(x))                  # size = (N, 32, W, H)
        score = self.classifier(x)                                # size = (N, 1, W, H)
        score = self.sigmoid(score)

        return score
        
class U_net32(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpooling =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu    = nn.LeakyReLU(inplace=True)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4   = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5   = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.classifier = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.maxpooling(self.relu(self.conv1(x)))  # size = (N, 64, W/2, H/2)
        x2 = self.maxpooling(self.relu(self.conv2(x1))) # size = (N, 128, W/4, H/4)
        x3 = self.maxpooling(self.relu(self.conv3(x2))) # size = (N, 256, W/8, H/8)
        x4 = self.maxpooling(self.relu(self.conv4(x3))) # size = (N, 512, W/16, H/16)
        x5 = self.maxpooling(self.relu(self.conv5(x4))) # size = (N, 1024, W/32, H/32)

        x = self.relu(self.deconv1(x5))                 # size = (N, 512, W/16, H/16)
        x = torch.cat((x4, x), 1)                       # size = (N, 1024, W/16, H/16)

        x = self.relu(self.deconv2(x))                  # size = (N, 256, W/8, H/8)
        x = torch.cat((x3, x), 1)                       # size = (N, 512, W/8, H/8)

        x = self.relu(self.deconv3(x))                  # size = (N, 128, W/4, H/4)
        x = torch.cat((x2, x), 1)                       # size = (N, 256, W/4, H/4)

        x = self.relu(self.deconv4(x))                  # size = (N, 64, W/2, H/2)
        x = torch.cat((x1, x), 1)                       # size = (N, 128, W/2, H/2)

        x = self.relu(self.deconv5(x))                  # size = (N, 32, W, H)
        score = self.classifier(x)                      # size = (N, 1, W, H)

        return score

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        print(pretrained)

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    batch_size, n_class, h, w = 10, 20, 160, 160

    # test output size
    vgg_model = VGGNet(requires_grad=True)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))
    output = vgg_model(input)
    assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])

    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    print("Pass size check")

    # test a random batch, loss should decrease
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(input)
        output = nn.functional.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        print("iter{}, loss {}".format(iter, loss.data[0]))
        optimizer.step()
