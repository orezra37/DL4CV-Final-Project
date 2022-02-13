import torch
from our_models.OG_former import OG1
from train_naive import train_naive as train
from OGDataset import OGDataset

num_heads = 4
lr = 1e-2
epochs = 10
batch_size = 4

my_dataset = OGDataset("data")
print('length of dataset:', len(my_dataset))
x, y = my_dataset[0]
s, z = x
print('S sample:', s.size(),
      '\nz sample:', z.size(),
      '\naatype sample:', y.size())

# naive_model_1 = OG1(num_heads=num_heads)
# train(model=naive_model_1,
#       batch_size=batch_size,
#       num_heads=num_heads,
#       lr=lr,
#       epochs=epochs)

# path = "Naive_model_state_dict"
# print(path)
# torch.save(naive_model_1.state_dict(), path)
