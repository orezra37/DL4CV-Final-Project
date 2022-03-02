import json
import torch
from our_models.OG_former import OGDefaultTransformer, OGOriginalTransformer
from train_naive import train_naive as train
from OGDataset import OGDataset

f = open('config.json')
conf = json.load(f)

num_heads = conf["num_heads"]
lr = conf["lr"]
epochs = conf["epochs"]
batch_size = conf["batch_size"]
test_every = conf["test_every"]
num_encoder_layers = conf["num_encoder_layers"]
num_decoder_layers = conf["num_decoder_layers"]
train_data_path = conf["train_data_path"]
test_data_path = conf["test_data_path"]

print(conf)

train_dataset = OGDataset(train_data_path)
test_dataset = OGDataset(test_data_path)

print('length of train dataset:', len(train_dataset), '\nlength of test dataset:', len(test_dataset))

print(train_dataset.device)

naive_model_1 = OGDefaultTransformer(num_heads=num_heads,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers)
naive_model_2 = OGOriginalTransformer(num_heads=num_heads)

train(model=naive_model_1,
      batch_size=batch_size,
      lr=lr,
      epochs=epochs,
      train_dataset=train_dataset,
      test_dataset=test_dataset,
      test_every=test_every)

# path = "Naive_model_state_dict"
# print(path)
# torch.save(naive_model_1.state_dict(), path)
