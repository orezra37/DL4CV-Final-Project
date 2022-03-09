import json
import torch
from our_models.OG_former import *
from OGDataset import OGDataset
from train_reverse import Trainer

config_path = 'config_reverse.json'
conf = json.load(open(config_path))

reversed_model = ReverseOriginalOG(config_path=config_path)
trainer = Trainer(config_path, reversed_model)

for key in conf:
    print('\n', key+':', conf[key])
print(
    '\nlength of train dataset:', len(trainer.train_loader),
    '\nlength of test dataset:', len(trainer.test_loader),
    '\n' + trainer.device
)

trainer.train()
