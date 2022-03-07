import json
import torch
from our_models.OG_former import OGDefaultTransformer, OGOriginalTransformer, ReverseOriginalOG
from OGDataset import OGDataset
from train_reverse_new import train_naive as train
from train_reverse import Trainer

config_path = 'config_reverse.json'
conf = json.load(open(config_path))

# lr = conf['lr']
# epochs = conf['epochs']
# train_dataset = OGDataset(conf['train_data_path'])
# test_dataset = OGDataset(conf['test_data_path'])
# test_every = conf['test_every']

reversed_model = ReverseOriginalOG(config_path=config_path)
trainer = Trainer(config_path, reversed_model)

# print(
#     conf,
#     '\nlength of train dataset:', len(train_dataset),
#     '\nlength of test dataset:', len(test_dataset),
#     '\n' + test_dataset.device
# )


print(
    conf,
    '\nlength of train dataset:', len(trainer.train_loader),
    '\nlength of test dataset:', len(trainer.test_loader),
    '\n' + trainer.device
)

trainer.train()

# train(model=reversed_model,
#       lr=lr,
#       epochs=epochs,
#       train_dataset=train_dataset,
#       test_dataset=test_dataset,
#       test_every=test_every)
