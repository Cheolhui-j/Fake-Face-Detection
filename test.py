from config import get_config
from trainer import trainer
from dataset.dataset import *
from validate import *

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
val, far, frr, acc = validate(model, testloader)

print(acc)