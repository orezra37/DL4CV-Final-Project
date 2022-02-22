import torch
from our_models.OG_former import OG
from train_naive import train_naive as train
from OGDataset import OGDataset

num_heads = 4
lr = 1e-2
epochs = 50
batch_size = 1
test_every = 5
num_encoding_layers=1

my_dataset = OGDataset("data")
print('length of dataset:', len(my_dataset))

naive_model_1 = OG(num_heads=num_heads, num_encoding_layers=num_encoding_layers)

train(model=naive_model_1,
      batch_size=batch_size,
      lr=lr,
      epochs=epochs,
      train_dataset=my_dataset,
      test_every=test_every)

# path = "Naive_model_state_dict"
# print(path)
# torch.save(naive_model_1.state_dict(), path)
