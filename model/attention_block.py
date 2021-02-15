import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Attention_block(nn.Module):
######### Image Attention Model #########
### Block 1 ###
    def __init__(self, input_channels, **kwargs):
        super(Attention_block, self).__init__()
        self.input_channels = input_channels

        self.depthwise_separable_conv_1 = nn.Conv2d(self.input_channels, 32, kernel_size=1, stride=1)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU(inplace=True)
        self.Attention_1 = Attention(32, 32)

        self.depthwise_separable_conv_2 = nn.Conv2d(32, 32*2, kernel_size=1, stride=1)
        self.batch_norm_2 = nn.BatchNorm2d(32*2)
        self.relu_2 = nn.ReLU(inplace=True)
        self.Attention_2 = Attention(32 * 2, 32 * 2)

        self.depthwise_separable_conv_3 = nn.Conv2d(32*2, 32*3, kernel_size=1, stride=1)
        self.batch_norm_3 = nn.BatchNorm2d(32*3)
        self.relu_3 = nn.ReLU(inplace=True)
        self.Attention_3 = Attention(32*3, 32*3)

        ### final stage ###
        self.conv_f = nn.Conv2d(32*3, 512, kernel_size=1, stride=1, bias=False) #origin - 576
        self.batch_norm_f = nn.BatchNorm2d(512)
        self.relu_f = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x1 = x

        x1 = self.depthwise_separable_conv_1(x1)
        x1 = self.batch_norm_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.Attention_1(x1)

        x2 = self.depthwise_separable_conv_2(x1)
        x2 = self.batch_norm_2(x2)
        x2 = self.relu_2(x2)
        x2 = self.Attention_2(x2)

        x3 = self.depthwise_separable_conv_3(x2)
        x3 = self.batch_norm_3(x3)
        x3 = self.relu_3(x3)
        x3 = self.Attention_3(x3)

        ### final stage ###
        x6 = self.conv_f(x3)
        x6 = self.batch_norm_f(x6)
        x6 = self.relu_f(x6)
        x6 = self.gap(x6)

        return x6.squeeze()


class Attention(nn.Module):
    def __init__(self, input_dim, out_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = input_dim
        self.out_dim = out_dim
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

        self.gamma = Variable(torch.zeros((1)), requires_grad=True).to('cuda')
        self.conv_f = nn.Conv2d(self.channels, self.filters_f_g, kernel_size=1, stride=1,bias=True)
        self.conv_g = nn.Conv2d(self.channels, self.filters_f_g, kernel_size=1, stride=1, bias=True)
        self.conv_h = nn.Conv2d(self.channels, self.filters_h, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        def hw_flatten(x):
            return x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        f = self.conv_f(x)  # [bs, c', h, w]
        g = self.conv_g(x)  # [bs, c', h, w]
        h = self.conv_h(x) # [bs, c, h, w]

        s = torch.bmm(hw_flatten(f).permute((0,2,1)), hw_flatten(g))  # # [bs, N, N] => N : h * w

        beta = F.softmax(s, dim=-1)  # attention map

        # plt.imshow(beta.cpu().detach().numpy()[0])
        # plt.show()

        o = torch.bmm(beta, hw_flatten(h).permute((0,2,1))).permute((0,2,1))  # [bs, N, C]

        o = o.view(x.shape)  # [bs, C, h, w]

        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape