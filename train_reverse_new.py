import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
import json


def train_naive(model, lr, epochs, train_dataset, test_dataset, test_every):
    """Train loop for OG-former based on a naive architecture."""
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(train_dataset.device)
    # model.device = train_dataset.device
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)
    losses = []
    accs = []
    max_accuracy = 0

    if epochs == "inf":
        epoch = 1
        while True:
            max_accuracy = run_epoch(model, epoch, criterion, train_loader, test_loader, optimizer, losses, accs,
                                     test_every, max_accuracy)
            epoch += 1
    else:
        for epoch in range(1, 1 + int(epochs)):
            max_accuracy = run_epoch(model, epoch, criterion, train_loader, test_loader, optimizer, losses, accs,
                                     test_every, max_accuracy)


def run_epoch(model, epoch, criterion, train_loader, test_loader, optimizer, losses, accs, test_every, max_accuracy):
    print('\nepoch:', epoch)
    print('train loss:', train_epoch(model, optimizer, train_loader, criterion))

    if epoch % test_every == 0:
        loss, accuracy = test_epoch(model, test_loader, criterion)
        print('test loss:', loss)
        print('accuracy', accuracy)
        losses.append(loss)
        accs.append(accuracy)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), model.model_name)
            print("saved!")
        torch.save((losses, accs), 'learning_statistics')
    return max_accuracy


def train_epoch(model, optimizer, train_loader, criterion):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimizer.zero_grad()
    model.train()
    model = model.to(device)
    running_loss = 0
    for i_batch, batch in enumerate(train_loader):
        x, y, _ = batch
        s = x[0]
        seq = y
        s, seq = s.to(device), seq.to(device)
        prediction = model(seq)
        loss = criterion(prediction, s)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def test_epoch(model, test_loader, criterion):
    running_loss = 0
    accuracy = 0
    for i_batch, batch in enumerate(test_loader):
        x, y, _ = batch
        s = x[0][0]
        seq = y
        prediction = model(seq)
        loss = criterion(prediction, s)
        accuracy += 100 * (1 - ((prediction - s).abs() / (prediction ** 2 + s ** 2) ** 0.5).mean())
        running_loss += loss.item()
    accuracy /= len(test_loader)
    return running_loss, accuracy.round().item()
