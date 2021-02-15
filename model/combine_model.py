from model.resnet import *
from model.attention_block import *
from model.gram_block import *
import torch.nn as nn

class combine_model(nn.Module):
    def __init__(self, mode, device):
        super(combine_model, self).__init__()
        self.base_model = None 
        self.net_mode = mode
        if self.net_mode == 'resnet':
            self.base_model = resnet18(pretrained = True)
        elif self.net_mode == 'gramnet':
            self.base_model = resnet18(pretrained = True)
            self.gram_blocki = GramBlock(3)
            self.gram_block1 = GramBlock(64)
            self.gram_block2 = GramBlock(64)
            self.gram_block3 = GramBlock(128)
            self.gram_block4 = GramBlock(256)
        elif self.net_mode == 'combine':
            self.base_model = resnet18(pretrained = True)
            self.gram_blocki = GramBlock(3)
            self.gram_block1 = GramBlock(64)
            self.gram_block2 = GramBlock(64)
            self.gram_block3 = GramBlock(128)
            self.gram_block4 = GramBlock(256)
            self.attention_block = Attention_block(3)
        else :
            print('No Model')

        self.dense_base = nn.Linear(512, 2).to(device)
        self.dense = nn.Linear(672, 2).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

    def forward(self, x):

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
            out = torch.cat((out_x,out_gi,out_g1,out_g2,out_g3,out_g4),1)
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



