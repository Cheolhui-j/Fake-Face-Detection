import os
import torch
from torch import optim
import torch.nn as nn
from model.combine_model import *
from model.resnet import *
from validate import *
from tensorboardX import SummaryWriter

class trainer(object):
    def __init__(self, conf):

        self.conf = conf

        if not os.path.exists(self.conf.work_path):
            os.makedirs(self.conf.work_path)
        if not os.path.exists(self.conf.model_path):
            os.makedirs(self.conf.model_path)
        if not os.path.exists(self.conf.log_path):
            os.makedirs(self.conf.log_path)

        self.model = None
        if self.conf.mode is not None:
            self.model = combine_model(self.conf.mode, self.conf.device)
        else :
            print('Model is None')

        self.model.to(conf.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=conf.lr, weight_decay=1e-4)

        self.writer = SummaryWriter(conf.log_path)

    def save_networks(self, epoch, optimizer, total_steps):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.conf.model_path, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'total_steps' : total_steps,
        }

        torch.save(state_dict, save_path)

    def load_state(self, epoch = 'best'):
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.conf.model_path, load_filename)           
        self.model.load_state_dict(torch.load(load_path)['model'])

    def evaluate(self, dataloader):
        if dataloader is None:
            print('Data is Empty')
        tar, far, frr, acc = validate(self.model, dataloader)
        return tar, far, frr, acc

    def save_bad_ex(self, dataset=None, dataloader = None):
        if dataloader is None or dataset is None:
            print('Data is Empty')
        tar, far, frr, acc = save_bad_ex(self.model, dataset, dataloader)
        return tar, far, frr, acc

    def adjust_learning_rate(self,optimizer,epoch):
        if epoch<self.conf.milestones[0]:
            lr=0.001
        elif epoch>=self.conf.milestones[0] and epoch<self.conf.milestones[1]:
            lr=0.0001
        elif epoch>=self.conf.milestones[1]:
            lr=0.00001
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

    def get_lr(self,optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr+=[param_group['lr']]
        return lr

    def train(self, dataloaders, testloader):

        self.model.train()
        running_loss = 0
        total_steps = 0
        maxi = 0

        for epoch in range(self.conf.epochs):
            self.adjust_learning_rate(self.optimizer,epoch)
            print (self.get_lr(self.optimizer))
            self.model.train()
            for inputs,labels in dataloaders:
                total_steps += 1
                inputs, labels = inputs.to(self.conf.device), labels.to(self.conf.device)
                self.optimizer.zero_grad()
                predict = self.model.forward(inputs)
                loss = self.criterion(predict, labels)
                #print (loss,'loss',epoch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if total_steps % self.conf.loss_freq == 0:
                    print("Train loss: {} at step: {}".format(loss, total_steps))
                    self.writer.add_scalar('loss', loss, total_steps)

                    print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                        ('', epoch, total_steps))
                    self.save_networks('latest', self.optimizer, total_steps)

            if epoch % self.conf.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
                self.save_networks('latest', self.optimizer, total_steps)
                self.save_networks(epoch, self.optimizer, total_steps)


            self.model.eval()
            val, far, frr, acc = self.evaluate(testloader)

            self.writer.add_scalar('val', val, total_steps)
            self.writer.add_scalar('far', far, total_steps)
            self.writer.add_scalar('frr', frr, total_steps)
            self.writer.add_scalar('acc', acc, total_steps)

            print ('epoch : {}, acc : {}, far : {}, frr : {}, acc : {}'.format(epoch, val, far, frr, acc))

            if acc>maxi:
                print('{} changed to {}'.format(maxi, acc))
                maxi=acc
                self.save_networks('best', self.optimizer, total_steps)
                self.save_networks(epoch, self.optimizer, total_steps)


