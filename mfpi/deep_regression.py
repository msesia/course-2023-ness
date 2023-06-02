import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
import numpy as np

from mfpi.utils import RegressionDataset

import warnings
import pdb

class NNet(nn.Module):
    """ Conditional mean estimator, formulated as neural net
    """
    def __init__(self, num_features, num_hidden=64, dropout=0.1):
        """ Initialization
        Parameters
        ----------
        num_features : integer, input signal dimension (p)
        num_hidden : integer, hidden layer dimension
        dropout : float, dropout rate
        """
        super(NNet, self).__init__()

        # Construct base network
        self.base_model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden,1)
        )
        self.init_weights()

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        y = self.base_model(x).flatten()
        return y



class Net:
    """ Fit a neural network (conditional mean) to training data
    """
    def __init__(self, num_features, dropout=0.2, learning_rate=0.001,
                 num_epochs=100, batch_size=16, num_hidden=64, random_state=0, calibrate=0, progress=True, verbose=False):
        """ Initialization
        Parameters
        ----------
        num_features : integer, input signal dimension (p)
        learning_rate : learning rate
        random_state : integer, seed used in CV when splitting to train-test
        """

        # Detect whether CUDA is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store input
        self.num_features = num_features
        self.progress = progress
        self.verbose = verbose

        # Define NNet model
        self.model = NNet(self.num_features, num_hidden=num_hidden, dropout=dropout)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize loss function
        self.loss_func = nn.MSELoss()

        # Store variables
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.calibrate = int(calibrate)

        # Initialize training logs
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

        # Validation
        self.val_period = 10


    def fit(self, X, Y, return_loss=False):
        Y = Y.flatten().astype(np.float32)
        X = X.astype(np.float32)

        dataset = RegressionDataset(X, Y)
        num_epochs = self.num_epochs
        if self.calibrate>0:
            if self.progress:
                print("Pre-training with 80% of the data to assess early stopping.".format(num_epochs))
            # Train with 80% of samples
            n_valid = int(np.round(0.2*X.shape[0]))
            loss_stats = []
            for b in range(self.calibrate):
                X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=n_valid, random_state=self.random_state+b)
                train_dataset = RegressionDataset(X_train, Y_train)
                val_dataset = RegressionDataset(X_valid, Y_valid)
                loss_stats_tmp = self._fit(train_dataset, num_epochs, val_dataset=val_dataset)
                loss_stats.append([loss_stats_tmp['val']])
                # Reset model
                self.model.init_weights()

            loss_stats = np.matrix(np.concatenate(loss_stats,0)).T

            loss_stats = np.median(loss_stats,1).flatten()
            # Find optimal number of epochs
            num_epochs = self.val_period*(np.argmin(loss_stats)+1)
            loss_stats_cal = loss_stats
            if self.progress:
                print("Selected number of epochs: {:d}. Training again using all the data...".format(num_epochs))

        # Train with all samples
        loss_stats = self._fit(dataset, num_epochs)
        if self.calibrate:
            loss_stats = loss_stats_cal

        if return_loss:
          return loss_stats

    def _fit(self, train_dataset, num_epochs, val_dataset=None):
        batch_size = self.batch_size

        # Initialize data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        if val_dataset is not None:
            val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        num_samples, num_features = train_dataset.X_data.shape
        if self.verbose:
            print("Training with {} samples and {} features.". \
                  format(num_samples, num_features))

        loss_stats = {'train': [], "val": []}

        X_train_batch = train_dataset.X_data.to(self.device)
        y_train_batch = train_dataset.y_data.to(self.device)

        if self.progress:
            e_iterator = tqdm(range(1, num_epochs+1))
        else:
            e_iterator = range(1, num_epochs+1)

        for e in e_iterator:

            # TRAINING
            train_epoch_loss = 0
            self.model.train()

            if batch_size<500:

              for X_train_batch, y_train_batch in train_loader:
                  X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                  self.optimizer.zero_grad()

                  y_train_pred = self.model(X_train_batch).to(self.device)
                  train_loss = self.loss_func(y_train_pred, y_train_batch)

                  train_loss.backward()
                  self.optimizer.step()

                  train_epoch_loss += train_loss.item()

            else:
                self.optimizer.zero_grad()

                y_train_pred = self.model(X_train_batch).to(self.device)

                train_loss = self.loss_func(y_train_pred, y_train_batch)

                train_loss.backward()
                self.optimizer.step()

                train_epoch_loss += train_loss.item()

            # VALIDATION
            if val_dataset is not None:
                if e % self.val_period == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_epoch_loss = 0
                        for X_val_batch, y_val_batch in val_loader:
                            X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                            y_val_pred = self.model(X_val_batch).to(self.device)
                            val_loss = self.loss_func(y_val_pred, y_val_batch)
                            val_epoch_loss += val_loss.item()

                    loss_stats['val'].append(val_epoch_loss/len(val_loader))
                    self.model.train()

            else:
                loss_stats['val'].append(0)

            if e % self.val_period == 0:
                loss_stats['train'].append(train_epoch_loss/len(train_loader))

            if (e % 10 == 0) and (self.verbose):
                if val_dataset is not None:
                    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | ', end='')
                    print(f'Val Loss: {val_epoch_loss/len(val_loader):.5f} | ', flush=True)
                else:
                    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | ', flush=True)

        return loss_stats

    def predict(self, X):
        """ Estimate the label given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of predicted labels (n)
        """
        self.model.eval()
        ret_val = self.model(torch.from_numpy(X).to(self.device).float().requires_grad_(False))
        return ret_val.cpu().detach().numpy()
