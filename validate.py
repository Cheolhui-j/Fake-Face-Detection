import numpy as np
import torch
import torch.nn
import cv2
import random,glob
from torch.autograd import Variable
from torchvision.utils import save_image

def validate(model, dataloader = None):
  model.eval()
  corr = 0
  if dataloader is None:
      print('[Test] Dataloader is Empty')
      raise dataloader

  pred = np.zeros((len(dataloader), 1))
  labels = np.zeros((len(dataloader), 1))

  for i, (inputs, label) in enumerate(dataloader):
     image_tensor =torch.tensor(inputs).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred[i] = np.argmax(output)
     labels[i] = label
     if pred[i] == labels[i]:
         corr += 1

  true_accept = np.sum(np.logical_and(pred, labels))
  false_accept = np.sum(np.logical_and(pred, np.logical_not(labels)))
  false_reject = np.sum(np.logical_and(np.logical_not(pred), labels))
  n_same = np.sum(labels)
  n_diff = np.sum(np.logical_not(labels))
  tar = float(true_accept) / float(n_same)
  far = float(false_accept) / float(n_diff)
  frr = float(false_reject) / float(n_same)

  return tar, far, frr, corr/len(dataloader)


def save_bad_ex(model, dataset=None, dataloader = None):
  model.eval()
  corr = 0
  if dataloader is None:
      print('[Test] Dataloader is Empty')
      raise dataloader

  pred = np.zeros((len(dataloader), 1))
  labels = np.zeros((len(dataloader), 1))

  for i, (inputs, label) in enumerate(dataloader):
     image_tensor =torch.tensor(inputs).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred[i] = np.argmax(output)
     labels[i] = label
     if pred[i] == labels[i]:
         corr += 1

  true_accept = np.sum(np.logical_and(pred, labels))
  false_accept = np.sum(np.logical_and(pred, np.logical_not(labels)))
  false_reject = np.sum(np.logical_and(np.logical_not(pred), labels))
  n_same = np.sum(labels)
  n_diff = np.sum(np.logical_not(labels))
  val = float(true_accept) / float(n_same)
  far = float(false_accept) / float(n_diff)
  frr = float(false_reject) / float(n_same)

  permute = [2, 1, 0]


  tar_list = list(np.where(np.logical_and(pred, labels))[0])
  for i in tar_list:
      save_image(dataset[i][0][permute, :]/255, './verification/tar/{}.jpg'.format(str(i)))
  far_list = list(np.where(np.logical_and(pred, np.logical_not(labels)))[0])
  for i in far_list:
      save_image(dataset[i][0][permute, :]/255, './verification/far/{}.jpg'.format(str(i)))
  frr_list = list(np.where(np.logical_and(np.logical_not(pred), labels))[0])
  for i in frr_list:
      save_image(dataset[i][0][permute, :]/255, './verification/frr/{}.jpg'.format(str(i)))
  trr_list = list(np.where(np.logical_and(np.logical_not(pred), np.logical_not(labels)))[0])
  for i in trr_list:
      save_image(dataset[i][0][permute, :]/255, './verification/trr/{}.jpg'.format(str(i)))

  return val, far, frr, corr/len(dataloader)
