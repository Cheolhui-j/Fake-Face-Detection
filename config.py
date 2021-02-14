from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
import os

def get_config(training = True):
    conf = edict()

    # Data Path
    conf.train.img_path = './data'
    conf.train.txt_path = './data'
    
    conf.test.img_path = './data'
    conf.test.txt_path = './data'
    
    conf.mode = 'combine' # resnet, gramnet, combine

    # Save Path
    conf.work_path = './workspace'
    conf.model_path = os.path.join(conf.work_path, 'models')
    conf.log_path = os.path.join(conf.work_path, 'logs')
    conf.save_path = os.path.join(conf.work_path, 'save')

    # enable cuda?
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    conf.batch_size = 32

    conf.epochs = 2000
    
    conf.train_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    conf.test_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    conf.lr = 1e-2
    conf.milestones = [50,150]

    return conf
