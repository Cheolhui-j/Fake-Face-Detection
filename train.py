from trainer import trainer
import os
from torch import nn
import torch.utils.model_zoo as model_zoo

from dataset import *
from config import get_config


if __name__ == '__main()__' :

    conf = get_config()

    image_datasets = customData(img_path=conf.train.img_path,txt_path=(conf.train.txt_path),
                                        data_transforms=conf.train_preprocess)

    dataloders =  torch.utils.data.DataLoader(image_datasets,
                                                    batch_size=conf.batch_size,
                                                    shuffle=True) 

    
    test_datasets = customData(img_path=conf.test.img_path,txt_path=(conf.test.txt_path),
                                        data_transforms=conf.test_preprocess)

    testloader =  torch.utils.data.DataLoader(test_datasets,
                                                    batch_size=1,
                                                    shuffle=False) 

    trainer = trainer(conf)

    trainer.train(dataloders, testloader)