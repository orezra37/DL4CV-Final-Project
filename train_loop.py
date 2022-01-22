import argparse
import yaml
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import LatentDataset
import EvoEncoder

def train():

    train_dataset = LatentDataset("/path/to/training/data")
    val_dataset = LatentDataset("/path/to/validation/data")
    test_dataset = LatentDataset("/path/to/test/data")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset)
    test_loader = DataLoader(test_dataset)

    criterion = CrossEntropyLoss()
    model = EvoEncoder(train=True)
    optimizer = Adam(model.parameters(), lr=config["lr"])

    for e in config['epoch_num']:

        model.train()

        for i_batch, batch in enumerate(train_loader):
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        for i_batch, batch in enumerate(val_loader):
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)



if __name__ == '__main__':

    p = argparse.ArgumentParser()
