import copy
import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np


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
        self.model.to(self.device)
        epoch = 1
        if self.conf['epochs'] == 'inf':
            while True:
                print('\nepoch:', epoch)
                self.train_epoch()
                if epoch % self.conf['test_every'] == 0:
                    self.test_epoch()
                    self.save_fig()
                epoch += 1
        else:
            while epoch <= self.conf['epochs']:
                print('\nepoch:', epoch)
                self.train_epoch()
                if epoch % self.conf['test_every'] == 0:
                    pred, s = self.test_epoch()
                epoch += 1
            self.save_fig()
            fig, ax = plt.subplots(2, 1)
            elem = np.arange(len(pred[0]))
            ax[0].plot(elem, pred[0].cpu().detach().numpy(), label='pred')
            ax[0].plot(elem, s[0][0].cpu().detach().numpy(), label='s')
            plt.legend()
            ax[1].plot(elem, (s[0]-pred)[0].abs().cpu().detach().numpy())
            plt.savefig('compare s and prediction')

    def train_epoch(self):
        self.optimizer.zero_grad()
        running_loss = 0
        for i_batch, batch in enumerate(self.train_loader):
            seq, s = self.model_forward(batch)
            prediction = self.model(seq, s)
            loss = self.criterion(prediction, s[0])
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        print('train loss:', running_loss)

    def test_epoch(self):
        running_loss = 0
        accuracy = 0
        self.model = self.model.to(self.device)
        for i_batch, batch in enumerate(self.test_loader):
            seq, s = Trainer.model_forward(batch)
            prediction = self.model(seq, s)
            running_loss = self.criterion(prediction, s[0]).item()
            accuracy += Trainer.accuracy_evaluation(prediction=prediction,
                                                    tgt=s)

        accuracy /= len(self.test_loader)
        self.loss_lst.append(running_loss)
        self.accuracy_lst.append(accuracy.item())
        print('test loss:', running_loss)
        print('accuracy', accuracy.round().item())

        if accuracy.item() < self.best_accuracy:
            self.best_accuracy = accuracy.item()
            if self.conf['model_save_name'] != 'None':
                torch.save(self.model.state_dict(), self.conf['model_save_name'])

        if self.conf['trainer_save_path'] != 'None': self.save_trainer()
        return prediction, s

    @staticmethod
    def model_forward(batch):
        x, y, _ = batch
        s = x[0]
        seq = y
        return seq, s

    @staticmethod
    def accuracy_evaluation(prediction, tgt):
        return 100 * (1 - ((prediction - tgt).abs() / (prediction ** 2 + tgt ** 2) ** 0.5).mean())

    def save_trainer(self):
        file = open(self.conf['trainer_save_path'], 'wb')
        pickle.dump(self.__dict__, file)
        file.close()

    def save_fig(self):
        fig, ax = plt.subplots(2, 1)
        loss = np.array(self.loss_lst)
        accuracy = np.array(self.accuracy_lst)
        epochs = np.arange(len(loss))
        ax[0].plot(epochs, loss)
        plt.ylabel('Loss')
        ax[1].plot(epochs, accuracy)
        plt.ylabel('Accuracy')
        if self.conf['trainer_save_path'] != 'None':
            plt.savefig(self.conf['trainer_save_path'])
        plt.close(fig)


