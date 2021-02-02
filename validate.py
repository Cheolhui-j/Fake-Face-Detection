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
from grad_cam import *

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (64,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def test(model, dataloader = None):
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


def gradcam_test(model, img_path):

    target_layers = ["relu","layer1", "layer2", "layer3", "layer4"]
    target_class = 0  # 1 - real, 0 - fake
    target_class_name = fake if target_class == 0 else real 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = []
    image_path = img_path
    image_paths.append(image_path)

    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)


    model.eval()
    model.forward(images)

    gcam = GradCAM(model=model.base_model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)
        save_gradcam(
            filename=osp.join(
                './result',
                "{}-{}-gradcam-{}-{}.png".format(
                    0, 'resnet18', target_layer, target_class_name
                ),
            ),
            gcam=regions[0, 0],
            raw_image=raw_images[0],
        )