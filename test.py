import numpy as np
import torch
import torch.nn
import cv2
import random,glob
from torch.autograd import Variable
from torchvision.utils import save_image

import resnet_gram as resnet
from attention_layer import *
from dataset import *
from network import Attention_model


test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_datasets = customData(img_path='./dataset/stylegan2_val',txt_path=('./stylegan2_val.txt')
                                    ,data_transforms=test_preprocess)
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = resnet.resnet18(pretrained=False)
att_model = att(3)
model = Attention_model(base_model, att_model)
pretrained=torch.load('./weights/attention_gram/model_epoch_best.pth')
model.load_state_dict(pretrained['model'],strict=False)
model.to(device)
model.eval()
val, far, frr, acc = test(model, testloader)
# val, far, frr, acc = save_bad_ex(model, test_datasets, testloader)
# img_path = ' '
# gradcam_test(model, img_path)

print(acc)