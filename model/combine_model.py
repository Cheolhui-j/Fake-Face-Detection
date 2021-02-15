from model.resnet import resnet, save_networks
from model.attention_block import *
from model.gram_block import *
from validate import *

class combine_model(object):
    def __init__(self, mode):
        self.base_model = None 
        self.net_mode = mode
        if self.net_mode == 'resnet':
            self.base_model = resnet.resnet18(pretrained = True)
        elif self.net_mode == 'gramnet':
            self.base_model = resnet.resnet18(pretrained = True)
            self.gram_blocki = gram_block(3)
            self.gram_block1 = gram_block(64)
            self.gram_block2 = gram_block(64)
            self.gram_block3 = gram_block(128)
            self.gram_block4 = gram_block(256)
        elif self.net_mode == 'combine':
            self.base_model = resnet.resnet18(pretrained = True)
            self.gram_blocki = gram_block(3)
            self.gram_block1 = gram_block(64)
            self.gram_block2 = gram_block(64)
            self.gram_block3 = gram_block(128)
            self.gram_block4 = gram_block(256)
            self.attention_block = attention_block(3)
        else :
            print('No Model')

        self.dense_base = nn.Linear(512, 2)
        self.dense = nn.Linear(704, 2)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def train(self, x):

        if self.net_mode == 'resnet':
            out_x, out_xi, out_x1, out_x2, out_x3, out_x4 = self.base_model(x)
            out = self.dense_base(out_x)
            out = self.softmax(out)

            return out

        elif self.net_mode == 'gramnet':
            out_x, out_xi, out_x1, out_x2, out_x3, out_x4 = self.base_model(x)
            out_gi = self.gram_blocki(out_xi) 
            out_g1 = self.gram_block1(out_x1)
            out_g2 = self.gram_block2(out_x2)
            out_g3 = self.gram_block3(out_x3)
            out_g4 = self.gram_block4(out_x4)
            out = torch.cat((out,out_gi,out_g1,out_g2,out_g3,out_g4),1)
            out = self.dense(out)
            out = self.softmax(out)

            return out

        elif self.net_mode == 'combine':    
            out_x, out_xi, out_x1, out_x2, out_x3, out_x4 = self.base_model(x)
            out_gi = self.gram_blocki(out_xi) 
            out_g1 = self.gram_block1(out_x1)
            out_g2 = self.gram_block2(out_x2)
            out_g3 = self.gram_block3(out_x3)
            out_g4 = self.gram_block4(out_x4)
            out_att = self.attention_block(x)
            out = torch.add(out_x, out_att)
            out = torch.cat((out,out_gi,out_g1,out_g2,out_g3,out_g4),1)
            out = self.dense(out)
            out = self.softmax(out)

            return out



