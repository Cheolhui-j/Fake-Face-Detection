import torch
import torch.nn as nn

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()

        features = input.view(a, b, c * d)

        a= features.transpose(1,2)

        G = torch.bmm(features, a)

        plt.imshow(G.cpu().detach().numpy()[0])
        plt.show()

        G=G.unsqueeze(1)
        return G.div(b* c * d)

class GramBlock(nn.Module):
    def __init__(self, input_channel):

        self.input_channel = input_channel
        
        self.gram = GramMatrix()

        self.conv_interi = nn.Sequential(nn.Conv2d(self.input_channel,32, kernel_size=3, stride=1, padding=1,
                                bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.gi_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(16), nn.ReLU())
        
        self.gi_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                                bias=False),nn.BatchNorm2d(32), nn.ReLU())
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        gi=self.conv_interi(x3)

        gi=self.gram(gi)

        gi=self.gi_fc1(gi)
        gi=self.gi_fc2(gi)

        gi = self.avgpool(gi)
        gi = gi.view(gi.size(0), -1)

        return gi

