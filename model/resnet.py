import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

##################### define Gram Matrix ######################
class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()

        features = input.view(a, b, c * d)

        a= features.transpose(1,2)

        G = torch.bmm(features, a)

        # plt.imshow(G.cpu().detach().numpy()[0])
        # plt.show()

        G=G.unsqueeze(1)
        return G.div(b* c * d)
###############################################################

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, Gram = True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)


        #################### define gram layer #####################
        self.gram = GramMatrix()

        self.conv_interi = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.gi_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.gi_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.conv_inter0 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.g0_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.g0_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.conv_inter1 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.g1_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.g1_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())
        
        self.conv_inter2 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.g2_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.g2_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())
        
        self.conv_inter3 = nn.Sequential(nn.Conv2d(128,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.g3_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.g3_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.conv_inter4 = nn.Sequential(nn.Conv2d(256,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.g4_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.g4_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ###############################################################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        xi=x

        x1 = self.conv1(xi)
        x1 = self.bn1(x1)
        x2 = self.relu(x1)
        x3 = self.maxpool(x1)


        x4 = self.layer1(x3)
        x5 = self.layer2(x4)
        x6 = self.layer3(x5)
        x = self.layer4(x6)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ###################### forward gram layer #####################

        gi=self.conv_interi(xi)
        gi=self.gram(gi)
        gi=self.gi_fc1(gi)
        gi=self.gi_fc2(gi)
        gi = self.avgpool(gi)
        gi = gi.view(gi.size(0), -1)

        g0=self.conv_inter0(x2)
        g0=self.gram(g0)
        g0=self.g0_fc1(g0)
        g0=self.g0_fc2(g0)
        g0 = self.avgpool(g0)
        g0 = g0.view(g0.size(0), -1)

        g1=self.conv_inter1(x3)
        g1=self.gram(g1)
        g1=self.g1_fc1(g1)
        g1=self.g1_fc2(g1)
        g1 = self.avgpool(g1)
        g1 = g1.view(g1.size(0), -1)

        g2=self.conv_inter2(x4)
        g2=self.gram(g2)
        g2=self.g2_fc1(g2)
        g2=self.g2_fc2(g2)
        g2 = self.avgpool(g2)
        g2 = g2.view(g2.size(0), -1)

        g3=self.conv_inter3(x5)
        g3=self.gram(g3)
        g3=self.g3_fc1(g3)
        g3=self.g3_fc2(g3)
        g3 = self.avgpool(g3)
        g3 = g3.view(g3.size(0), -1)
        
        g4=self.conv_inter4(x6)
        g4=self.gram(g4)
        g4=self.g4_fc1(g4)
        g4=self.g4_fc2(g4)
        g4 = self.avgpool(g4)
        g4 = g4.view(g4.size(0), -1)

        ############################################################### 

        #x = self.fcnewr(x)

        return x, gi, g0, g1, g2, g3, g4


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model