from model.resnet_gram import resnet, save_networks
from model.attention_layer import *

class Attention_model(nn.Module):
    def __init__(self, base_model, att_model):
        super(Attention_model, self).__init__()
        self.base_model = base_model
        self.att_model = att_model
        self.dense = nn.Linear(704, 2)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out_x, out_gi, out_g0, out_g1, out_g2, out_g3, out_g4 = self.base_model(x)
        out_att = self.att_model(x)
        # out_x = self.avgpool(out_x)
        # out_x = out_x.view(x.size(0), -1)
        out = torch.add(out_x, out_att)
        out = torch.cat((out,out_gi,out_g0,out_g1,out_g2,out_g3,out_g4),1)
        out = self.dense(out)
        out = self.softmax(out)

        return out
