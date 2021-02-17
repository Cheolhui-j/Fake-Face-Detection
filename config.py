from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms 
import os

def get_config(training = True):
    conf = edict()

    # trian Data Path
    conf.train_img_path = './data/stylegan_train'
    conf.train_txt_path = './data/stylegan_train.txt'
    # test Data Path
    conf.test_img_path = './data/stylegan2_val'
    conf.test_txt_path = './data/stylegan2_val.txt'
    
    # model name
    conf.mode = 'combine' # resnet, gramnet, combine

    # Save model & tensorboard log Path
    conf.work_path = './workspace'
    conf.model_path = os.path.join(conf.work_path, 'models')
    conf.log_path = os.path.join(conf.work_path, 'logs')

    # check if cuda is available
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # batch size
    conf.batch_size = 16

    # epoch
    conf.epochs = 2000

    # It means the step to print the loss and the epoch to save the model, respectively.
    conf.loss_freq = 100
    conf.save_epoch_freq =10
    
    # Data argumentation
    conf.train_preprocess = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    conf.test_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # initialize learning rate and when to decay learning rate 
    conf.lr = 1e-2
    conf.milestones = [50,80,110]

    return conf
