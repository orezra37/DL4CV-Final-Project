import json
import torch
from our_models.OG_former import OG
from train_naive import train_naive as train
from OGDataset import OGDataset

f = open('config.json')
data = json.load(f)

num_heads = data["num_heads"]
lr = data["lr"]
epochs = data["epochs"]
batch_size = data["batch_size"]
test_every = data["test_every"]
num_encoder_layers = data["num_encoder_layers"]
num_decoder_layers = data["num_decoder_layers"]

print(data)

my_dataset = OGDataset("data")
print('length of dataset:', len(my_dataset))

naive_model_1 = OG(num_heads=num_heads, num_encoder_layers=0, num_decoder_layers=5)

train(model=naive_model_1,
      batch_size=batch_size,
      lr=lr,
      epochs=epochs,
      train_dataset=my_dataset,
      test_every=test_every)

# path = "Naive_model_state_dict"
# print(path)
# torch.save(naive_model_1.state_dict(), path)
