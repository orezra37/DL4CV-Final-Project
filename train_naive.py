import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from our_models.OG_former import OG
from torch.optim import Adam


def train_naive(model, criterion=CrossEntropyLoss(), lr=1e-2, batch_size=4, epochs=10, train_dataset=OGDataset("data"),
                test_every=10):
    """Train loop for OG-former based on a naive architecture."""
    model = model.to(train_dataset.device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    for e in range(1, 1 + epochs):
        if e % test_every == 0:
            print('epoch:', e)
        model.train()
        optimizer.zero_grad()
        running_loss = 0
        for i_batch, batch in enumerate(train_loader):
            x, y, _ = batch
            s, z = x
            probs = model(s, z)
            _, prediction = torch.max(probs.data, 2)
            loss = criterion(probs[0], y[0])
            correct = (prediction == y).sum()
            accuracy = torch.round(100 * correct / s.size(1)).item()
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        if e % test_every == 0:
            print('loss', running_loss)
            print('accuracy', accuracy, '\n')
