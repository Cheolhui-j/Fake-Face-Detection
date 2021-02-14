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

        self.model = None
        if self.conf.mode is not None:
            model = combine_model(self.conf.mode)
        else :
            print('Model is None')

        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

        self.writer = SummaryWriter(conf.log_path)

    def save_networks(model, epoch, optimizer, total_steps):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.conf.model_path, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'total_steps' : total_steps,
        }

        torch.save(state_dict, save_path)

    def load_state(self, save_path, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.conf.model_path, load_filename)           
        self.model.load_state_dict(torch.load(load_path))

    def evaluate(self, dataloader):
        tar, far, frr, acc = evaluate(self.model, dataloader)
        return tar, far, frr, acc

    def save_bad_ex(self, dataset=None, dataloader = None):
        tar, far, frr, acc = evaluate(self.model, dataset, dataloader)
        return tar, far, frr, acc

    def adjust_learning_rate(optimizer,epoch):
        if epoch<50:
            lr=0.001
        elif epoch>=50 and epoch<150:
            lr=0.0001
        elif epoch>=150:
            lr=0.00001
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

    def get_lr(optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr+=[param_group['lr']]
        return lr

    def train(self, dataloaders, testloader):

        self.model.train()
        loss = 0
        total_steps = 0

        for epoch in range(epochs):
            adjust_learning_rate(optimizer,epoch)
            print (get_lr(optimizer))
            self.model.train()
            for inputs,labels in dataloders:
                total_steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                predict = model.forward(inputs)
                loss = criterion(predict, labels)
                #print (loss,'loss',epoch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if total_steps % loss_freq == 0:
                    print("Train loss: {} at step: {}".format(loss, total_steps))
                    self.writer.add_scalar('loss', loss, total_steps)

                    print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                        ('', epoch, total_steps))
                    self.save_networks(model, 'latest', optimizer, total_steps)

                if epoch % save_epoch_freq == 0:
                    print('saving the model at the end of epoch %d, iters %d' %
                        (epoch, total_steps))
                    self.save_networks(model, 'latest', optimizer, total_steps)
                    self.save_networks(model, epoch, optimizer, total_steps)


            model.eval()
            val, far, frr, acc = self.evaluate(model, testloader)

            self.writer.add_scalar('val', val, total_steps)
            self.writer.add_scalar('far', far, total_steps)
            self.writer.add_scalar('frr', frr, total_steps)
            self.writer.add_scalar('acc', acc, total_steps)

            print ('epoch : {}, acc : {}, far : {}, frr : {}, acc : {}'.format(epoch, val, far, frr, acc))

            if acc>maxi:
                print('{} changed to {}'.format(maxi, acc))
                maxi=acc
                self.save_networks(model, 'best', optimizer, total_steps)
                self.save_networks(model, epoch, optimizer, total_steps)


