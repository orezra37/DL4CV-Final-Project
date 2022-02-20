import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from our_models.OG_former import OG
from torch.optim import Adam


def train_naive(model, lr=1e-2, batch_size=4, epochs=10, train_dataset=OGDataset("data"),
                test_every=10):
    """Train loop for OG-former based on a naive architecture."""
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(train_dataset.device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for e in range(1, 1 + int(epochs)):
        if e % test_every == 0:
            print('epoch:', e)
        optimizer.zero_grad()
        model.train()
        running_loss = 0
        for i_batch, batch in enumerate(train_loader):
            x, y, _ = batch
            s, z = x
            probs = model(s, z)
            loss = criterion(probs[i_batch], y[i_batch])
            prediction = torch.argmax(probs.data, dim=2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct = (prediction == y).sum()
            accuracy = torch.round(100 * correct / s.size(1)).item()
        if e % test_every == 0:
            print('loss', running_loss)
            print('accuracy', accuracy, '\n')
