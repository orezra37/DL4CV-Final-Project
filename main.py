from our_models.OG_former import OG1
from train_loop import train_naive as train

num_heads = 4
lr = 1e-2
epochs = 10
batch_size = 4

model = OG1(num_heads=num_heads)
train(model=model,
      batch_size=batch_size,
      num_heads=num_heads,
      lr=lr,
      epochs=epochs)