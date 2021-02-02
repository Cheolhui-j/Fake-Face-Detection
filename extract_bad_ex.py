import torch.utils.model_zoo as model_zoo
import resnet18_gram as resnet

from dataset import *
from validate import *

test_datasets = customData(img_path='./dataset/val',txt_path=('val.txt')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet.resnet18(pretrained=False)
pretrained=torch.load('/mnt/hdd_10tb_2/cheolhui_hdd_10tb_2/gramnet_blur/weights/model_epoch_latest.pth')
model.load_state_dict(pretrained['model'],strict=False)
model.to(device)

#model.eval()
val, far, frr, acc = save_bad_ex(model, test_datasets, testloader)
print('tar : {}, far : {}, frr : {}, acc : {}'.format(val, far, frr, acc))
