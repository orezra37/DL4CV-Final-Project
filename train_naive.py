import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def train_naive(model, lr, batch_size, epochs, train_dataset, test_every):
    """Train loop for OG-former based on a naive architecture."""
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(train_dataset.device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader
    max_accuracy = 0
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
            loss = criterion(probs[0], y[0])
            prediction = torch.argmax(probs.data, dim=2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct = (prediction == y).sum()
            accuracy = torch.round(100 * correct / s.size(1)).item()
        if e % test_every == 0:
            print('loss', running_loss)
            print('accuracy', accuracy, '\n')
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(model.state_dict(), "best_run")
