import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def train_naive(model, lr, batch_size, epochs, train_dataset, test_dataset, test_every):
    """Train loop for OG-former based on a naive architecture."""
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(train_dataset.device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    losses = []
    accs = []
    max_accuracy = 0

    for e in range(1, 1 + int(epochs)):
        print('\nepoch:', e)
        print('train loss:', train_epoch(model, optimizer, train_loader, criterion))

        if e % test_every == 0:
            loss, accuracy = test_epoch(model, test_loader, criterion)
            print('test loss:', loss)
            print('accuracy', accuracy)
            losses.append(loss)
            accs.append(accuracy)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(model.state_dict(), "best_run")
            torch.save((losses, accs), 'learning_statistics')

def train_epoch(model, optimizer, train_loader, criterion):
    optimizer.zero_grad()
    model.train()
    running_loss = 0
    for i_batch, batch in enumerate(train_loader):
        x, y, _ = batch
        s, z = x
        probs = model(s, z)
        loss = criterion(probs[0], y[0])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def test_epoch(model, test_loader, criterion):
    running_loss = 0
    accuracy = 0
    for i_batch, batch in enumerate(test_loader):
        x, y, _ = batch
        s, z = x
        probs = model(s, z)
        loss = criterion(probs[0], y[0])
        prediction = torch.argmax(probs.data, dim=2)
        correct = (prediction == y).sum()
        accuracy += (100 * correct / s.size(1))
        running_loss += loss.item()
    accuracy /= len(test_loader)
    return running_loss, accuracy.round().item()
