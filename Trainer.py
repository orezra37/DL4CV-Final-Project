import torch
from OGDataset import OGDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class Trainer:
    """
    Trainer class in order to train models.
    """

    def __init__(self, config_path, model=None):
        """
        Initializing the trainer.
        :param config_path: A jason format confing file.
        :param model: The model we would like to train. if None assume that keep training a previous loaded trainer.
        """
        self.conf = json.load(open(config_path))
        self.model = model

        if self.conf['trainer_load_path'] == 'None':
            self.loss_lst = []
            self.accuracy_lst = []
            self.epoch_lst = []
            self.best_accuracy = 0
            self.fig_lst = []
        else:
            file = open(self.conf['trainer_load_path'], 'rb')
            self.__dict__ = pickle.load(file)
            file.close()
            self.conf = json.load(open(config_path))

        if self.model is not None and self.conf['model_load_path'] != 'None':
            self.model = model
            self.model.load_state_dict(torch.load(self.conf['model_load_path']))

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), self.conf['lr'])
        self.train_loader = DataLoader(dataset=OGDataset(self.conf['train_data_path']),
                                       shuffle=True,
                                       batch_size=self.conf['batch_size'])
        self.test_loader = DataLoader(dataset=OGDataset(self.conf['test_data_path']),
                                      shuffle=True,
                                      batch_size=self.conf['batch_size'])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def train(self):
        """ Runs a general train loop"""
        self.model.train()
        self.model.to(self.device)
        epoch = 1
        num_epochs = float(self.conf['epochs'])

        while epoch <= num_epochs:
            print('\nepoch:', epoch)
            self.train_epoch()
            if epoch % self.conf['test_every'] == 0:
                self.test_epoch(epoch)
                self.save_fig(epoch)
            epoch += 1

    def train_epoch(self):
        """
        Trains a single epoch.
        """
        self.optimizer.zero_grad()
        running_loss = 0
        for i_batch, batch in enumerate(self.train_loader):
            x, prediction, tgt = self.model_forward(batch)
            loss = self.criterion(prediction, tgt)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        print('train loss:', running_loss)

    def test_epoch(self, epoch):
        """
        Tests epoch - saves the epoch number, loss and accuracy in their lists.
         Also prints figure for the statistics collected so far.
        """
        running_loss = 0
        accuracy = 0
        self.model = self.model.to(self.device)
        for i_batch, batch in enumerate(self.test_loader):
            x, prediction, tgt = self.model_forward(batch)
            running_loss += self.criterion(prediction, tgt).item()
            accuracy += Trainer.accuracy_evaluation(prediction, tgt)

        accuracy /= len(self.test_loader)
        self.epoch_lst.append(epoch)
        self.loss_lst.append(running_loss)
        self.accuracy_lst.append(accuracy.item())
        print('test loss:', running_loss)
        print('accuracy:', ((accuracy * 10).round() / 10).item())

        if accuracy.item() < self.best_accuracy:
            self.best_accuracy = accuracy.item()
            if self.conf['model_save_name'] != 'None':
                torch.save(self.model.state_dict(), self.conf['model_save_name'])

        if self.conf['trainer_save_path'] != 'None':
            self.save_trainer()

    def model_forward(self, batch):
        """
        Processing and forwarding the batch from the loader.
        :param batch: A batch of samples from the loader.
        :return: tuple of size 3 contains the samples, the prediction of the model, and the target label.
        """
        x, y, _ = batch
        s, z = x
        seq = y
        prediction = self.model(s, z)
        return s[0], prediction[0], seq[0]

    @staticmethod
    def accuracy_evaluation(prediction, tgt):
        """
        Calculating the accuracy for some prediction and target.
        """
        # return 100 * (1 - ((prediction - tgt).abs() / (prediction ** 2 + tgt ** 2) ** 0.5).mean())
        return 100 * (torch.argmax(prediction.data, dim=1) == tgt).sum() / prediction.size(1)

    def save_trainer(self):
        """
        Saves the trainer, including the model and the statistics collected so far.
        """
        file = open(self.conf['trainer_save_path'], 'wb')
        pickle.dump(self.__dict__, file)
        file.close()

    def save_fig(self, epoch):
        """
        Creating and saving a figure of the learning statistic collected every test epoch.
        :param epoch: The number of the epoch which is the figure was created.
        :return: creates a figure with epoch and time labels for the learning statistics so far.
        """
        fig, ax = plt.subplots(2, 1)
        loss = np.array(self.loss_lst)
        accuracy = np.array(self.accuracy_lst)
        epochs = np.array(self.epoch_lst)
        ax[0].plot(epochs, loss)
        plt.ylabel('Loss')
        ax[1].plot(epochs, accuracy)
        plt.ylabel('Accuracy')
        if self.conf['trainer_save_path'] != 'None':
            time_label = datetime.now().strftime("%d_%m_%H_%M_%S")
            plt.savefig(self.conf['figs_path'] + time_label + ' epoch_' + str(epoch))
            self.fig_lst.append((fig, ax))
        plt.close(fig)
