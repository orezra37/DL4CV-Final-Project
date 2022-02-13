from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from our_models.OG_former import OG1
from torch.optim import Adam


def train_naive(model, criterion=CrossEntropyLoss, lr=1e-2, batch_size=4, epochs=10):
    """Train loop for OG-former based on a naive architecture."""
    train_dataset = OGDataset("data")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        print(e)
        model.train()

        for i_batch, batch in enumerate(train_loader):
            x, y, _ = batch
            pred = model(x)
            loss = criterion(pred, y[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
