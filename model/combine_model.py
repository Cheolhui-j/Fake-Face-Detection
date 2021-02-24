from model.resnet import *
from model.attention_block import *
import torch.nn as nn

class combine_model(nn.Module):
    def __init__(self, device):
        super(combine_model, self).__init__()
        self.base_model = None 
        
        self.base_model = resnet18(pretrained = True).to(device)
        self.attention_block = Attention_block(3)

        self.dense = nn.Linear(704, 2).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

    def forward(self, x):
  
        out_x, out_gi, out_g0, out_g1, out_g2, out_g3, out_g4 = self.base_model(x)
        out_att = self.attention_block(x)
        out = torch.add(out_x, out_att)
        out = torch.cat((out,out_gi,out_g0,out_g1,out_g2,out_g3,out_g4),1)
        out = self.dense(out)
        out = self.softmax(out)

        return out



