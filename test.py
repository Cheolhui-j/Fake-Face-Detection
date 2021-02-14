import numpy as np
import torch
import torch.nn
import cv2
import random,glob
from torch.autograd import Variable
from torchvision.utils import save_image

from model.combine_model import *
from config import get_config
from trainer import trainer

conf = get_config()

test_datasets = customData(img_path=conf.test.img_path,txt_path=(conf.test.txt_path)
                                    ,data_transforms=conf.test_preprocess)
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False)


model = trainer(conf)
model.load_state()
model.eval()
val, far, frr, acc = test(model, testloader)

print(acc)