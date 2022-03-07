import copy
import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
import json
import pickle


class Trainer:
    """
    Trainer class in order to train models.
    The trainer is suitable for the current CNN.
    """

    def __init__(self, config_path, model):
        self.conf = json.load(open(config_path))
        if self.conf['trainer_load_path'] != 'None':
            file = open(self.conf['trainer_load_path'], 'rb')
            self.__dict__ = pickle.load(file)
            file.close()

        else:
            self.model = model
            if self.conf['model_load_path'] != 'None':
                self.model.load_state_dict(torch.load(self.conf['model_load_path']))
            self.loss_lst = []
            self.accuracy_lst = []
            self.best_accuracy = 10e6

        self.conf = json.load(open(config_path))
        self.criterion = MSELoss()
        self.optimizer = Adam(model.parameters(), self.conf['lr'])
        self.train_loader = DataLoader(OGDataset(self.conf['train_data_path']), shuffle=True)
        self.test_loader = DataLoader(OGDataset(self.conf['test_data_path']), shuffle=True)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def train(self):
        self.model.train()
        epoch = 1
        if self.conf['epochs'] == 'inf':
            while True:
                print('\nepoch:', epoch)
                self.train_epoch()
                if epoch % self.conf['test_every'] == 0:
                    self.test_epoch()
                epoch += 1
        else:
            while epoch <= self.conf['epochs']:
                print('\nepoch:', epoch)
                self.train_epoch()
                if epoch % self.conf['test_every'] == 0:
                    self.test_epoch()
                epoch += 1

    def train_epoch(self):
        self.optimizer.zero_grad()
        running_loss = 0
        model = copy.deepcopy(self.model.to(self.device))
        # model = self.model.to(self.device)
        for i_batch, batch in enumerate(self.train_loader):
            seq, s = self.model_forward(batch)
            seq, s = seq.to(self.device), s.to(self.device)
            prediction = model(seq)
            loss = self.criterion(prediction, s)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.loss_lst.append(running_loss)
        print('train loss:', running_loss)

    def test_epoch(self):
        running_loss = 0
        accuracy = 0
        self.model = self.model.to(self.device)
        for i_batch, batch in enumerate(self.test_loader):
            seq, s = Trainer.model_forward(batch)
            seq, s = seq.to(self.device), s.to(self.device)
            prediction = self.model(seq)
            running_loss = self.criterion(prediction, s).item()
            accuracy += Trainer.accuracy_evaluation(prediction=prediction,
                                                    tgt=s)

        accuracy /= len(self.test_loader)
        self.accuracy_lst.append(accuracy.item())
        print('test loss:', running_loss)
        print('accuracy', accuracy.round().item())

        if accuracy.item() < self.best_accuracy:
            self.best_accuracy = accuracy.item()
            if self.conf['model_save_name'] != 'None':
                torch.save(self.model.state_dict(), self.conf['model_save_name'])

        if self.conf['trainer_save_path'] != 'None': self.save_trainer()

    @staticmethod
    def model_forward(batch):
        x, y, _ = batch
        s = x[0][0]
        seq = y
        return seq, s

    @staticmethod
    def accuracy_evaluation(prediction, tgt):
        return 100 * (1 - ((prediction - tgt).abs() / (prediction ** 2 + tgt ** 2) ** 0.5).mean())

    def save_trainer(self):
        file = open(self.conf['trainer_save_path'], 'wb')
        pickle.dump(self.__dict__, file)
        file.close()
