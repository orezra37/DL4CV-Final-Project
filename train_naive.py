import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from our_models.OG_former import OG1
from torch.optim import Adam


def train_naive(model, criterion=CrossEntropyLoss(), lr=1e-2, batch_size=4, epochs=10):
    """Train loop for OG-former based on a naive architecture."""
    train_dataset = OGDataset("data")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        print('epoch:', e)
        model.train()

        for i_batch, batch in enumerate(train_loader):
            x, y, _ = batch
            s, z = x
            probs = model(s, z)
            _, pred = torch.max(probs.data, 2)
            loss = criterion(probs[0], y[0])
            correct = (pred == y).sum().item()
            accuracy = correct / s.size(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            print(accuracy)
