import json
import torch
from our_models.OG_former import OGOriginalTransformer, ReverseDefaultTransformerOG
from OGDataset import OGDataset
from Trainer import Trainer

config_path = 'config_final.json'
conf = json.load(open(config_path))

model = OGOriginalTransformer(num_heads=conf['num_heads'])
trainer = Trainer(config_path, model)

for key in conf:
    print('\n', key+':', conf[key])
print(
    '\nlength of train dataset:', len(trainer.train_loader),
    '\nlength of test dataset:', len(trainer.test_loader),
    '\n' + trainer.device
)

trainer.train()
