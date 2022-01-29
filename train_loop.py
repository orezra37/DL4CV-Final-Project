import argparse
import yaml
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from OGDataset import OGDataset
from our_models.EvoEncoder import EvoEncoder

def train(args):

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



if __name__ == '__main__':
    import pydevd_pycharm
    pydevd_pycharm.settrace('10.96.2.219', port=123, stdoutToServer=True, stderrToServer=True)
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=str)
    args = p.parse_args()
    train(args)
