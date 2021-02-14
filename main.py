from trainer import trainer
import os
from torch import nn
import torch.utils.model_zoo as model_zoo

from dataset import *


if __name__ == '__main()__' :

    conf = get_config()

    image_datasets = customData(img_path=conf.train.img_path,txt_path=(conf.train.txt_path),
                                        data_transforms=conf.train_preprocess)
                                        #dataset=x) for x in ['train', 'val']}

    dataloders =  torch.utils.data.DataLoader(image_datasets,
                                                    batch_size=conf.batch_size,
                                                    shuffle=True) 

    
    test_datasets = customData(img_path=conf.test.img_path,txt_path=(conf.test.txt_path),
                                        data_transforms=conf.test_preprocess)
                                        #dataset=x) for x in ['train', 'val']}

    testloader =  torch.utils.data.DataLoader(test_datasets,
                                                    batch_size=1,
                                                    shuffle=False) 
        
    conf = get_config()

    trainer = trainer(conf)

    trainer.train(conf, dataloders, testloader)

# import os
# from torch import nn
# from torch import optim
# import torch.utils.model_zoo as model_zoo
# from tensorboardX import SummaryWriter

# from dataset import *
# from validate import *
# from model.resnet_gram import save_networks
# from model.attention_layer import *
# from network import Attention_model

# train_preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# test_preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# image_datasets = customData(img_path='./dataset/pggan_train',txt_path=('pggan_train.txt'),
#                                     data_transforms=train_preprocess)
#                                     #dataset=x) for x in ['train', 'val']}

# dataloders =  torch.utils.data.DataLoader(image_datasets,
#                                                  batch_size=32,
#                                                  shuffle=True) 

 
# test_datasets = customData(img_path='./dataset/stylegan_val',txt_path=('stylegan_val.txt'),
#                                     data_transforms=test_preprocess)
#                                     #dataset=x) for x in ['train', 'val']}

# testloader =  torch.utils.data.DataLoader(test_datasets,
#                                                  batch_size=1,
#                                                  shuffle=False) 



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #model = resnet.resnet18(pretrained=True)
# base_model = resnet.resnet18(pretrained=True)
# att_model = att(3)
# model = Attention_model(base_model, att_model)

# #criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
# model.to(device)



# epochs = 300
# steps = 0
# running_loss = 0
# print_every = 1000
# train_losses, test_losses = [], []
# loss_freq = 100
# save_epoch_freq = 10

# train_writer = SummaryWriter(os.path.join('./checkpoints', 'resnet_att_gram', "train"))
# val_writer = SummaryWriter(os.path.join('./checkpoints', 'resnet_att_gram', "val"))

# maxi=0
# total_steps = 0
# for epoch in range(epochs):
#     adjust_learning_rate(optimizer,epoch)
#     print (get_lr(optimizer))
#     model.train()
#     for inputs,labels in dataloders:
#         total_steps += 1
#         model.train()
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         logps = model.forward(inputs)
#         loss = criterion(logps, labels)
#         #print (loss,'loss',epoch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#         if total_steps % loss_freq == 0:
#             print("Train loss: {} at step: {}".format(loss, total_steps))
#             train_writer.add_scalar('loss', loss, total_steps)

#             print('saving the latest model %s (epoch %d, model.total_steps %d)' %
#                   ('', epoch, total_steps))
#             save_networks(model, 'latest', optimizer, total_steps)

#     if epoch % save_epoch_freq == 0:
#         print('saving the model at the end of epoch %d, iters %d' %
#               (epoch, total_steps))
#         save_networks(model, 'latest', optimizer, total_steps)
#         save_networks(model, epoch, optimizer, total_steps)


#     model.eval()
#     val, far, frr, acc = test(model, testloader)

#     val_writer.add_scalar('val', val, total_steps)
#     val_writer.add_scalar('far', far, total_steps)
#     val_writer.add_scalar('frr', frr, total_steps)
#     val_writer.add_scalar('acc', acc, total_steps)

#     print ('epoch : {}, acc : {}, far : {}, frr : {}, acc : {}'.format(epoch, val, far, frr, acc))

#     if acc>maxi:
#         print('{} changed to {}'.format(maxi, acc))
#         maxi=acc
#         save_networks(model, 'best', optimizer, total_steps)
#         save_networks(model, epoch, optimizer, total_steps)




