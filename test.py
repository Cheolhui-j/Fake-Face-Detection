from config import get_config
from trainer import trainer
from dataset.dataset import *
from validate import *

conf = get_config()

test_datasets = customData(img_path=conf.test_img_path,txt_path=(conf.test_txt_path)
                                    ,data_transforms=conf.test_preprocess)
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False)


trainer = trainer(conf)
trainer.load_state()
trainer.model.eval()
val, far, frr, acc = validate(trainer.model, testloader)

print(acc)