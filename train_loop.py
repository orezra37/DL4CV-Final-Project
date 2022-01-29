import argparse
import yaml
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from OGDataset import OGDataset
from our_models.EvoEncoder import EvoEncoder
from our_models.OG_former import OG1


def train(args):
    """Train loop for OG-former based on EVO-former."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_dataset = OGDataset("data")
    # val_dataset = OGDataset("/path/to/validation/data")
    # test_dataset = OGDataset("/path/to/test/data")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_dataset)
    # test_loader = DataLoader(test_dataset)

    criterion = CrossEntropyLoss()
    model = EvoEncoder(train=True)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=float(config["lr"]))

    for e in range(config['epochs']):
        print(e)
        model.train()

        for i_batch, batch in enumerate(train_loader):
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

        # model.eval()

        # for i_batch, batch in enumerate(val_loader):
        #     x, y = batch
        #     pred = model(x)
        #     loss = criterion(pred, y)


def train_naive(model=None, batch_size=4, num_heads=4, lr=1e-2, epochs=10):
    """Train loop for OG-former based on a naive architecture."""
    train_dataset = OGDataset("data")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = CrossEntropyLoss()
    if model is None: model = OG1(num_heads=num_heads)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        print(e)
        model.train()

        for i_batch, batch in enumerate(train_loader):
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

if __name__ == '__main__':
    import pydevd_pycharm
    pydevd_pycharm.settrace('10.96.2.219', port=123, stdoutToServer=True, stderrToServer=True)
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=str)
    args = p.parse_args()
    train(args)
