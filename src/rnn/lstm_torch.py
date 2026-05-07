"""
LSTM
Modified from
https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=CKEzO1jzKydL
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    """Long-short-term memory model (LSTM)"""

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int):
        """Initialize LSTM"""
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # nn.RNN

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """LSTM forward propagation"""
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


def sliding_windows(data, seq_length):
    """Create sliding windows"""
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i: (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def load_data(dataset_path: str, dataset_label: str):
    """Load data"""
    training_set = pd.read_csv(dataset_path)
    training_set = training_set.iloc[:, 1: 2].values

    plt.plot(training_set, label=dataset_label)
    plt.title(f'Training Set ({dataset_label})')
    plt.show()

    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    seq_length = 4
    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return sc, train_size, test_size,  dataX, dataY,  trainX, trainY,   testX, testY


def train_and_test_model(dataset_path: str, dataset_label: str, params: dict):
    """Train and test"""
    if params['verbose'] is None or params['verbose'] is False:
        log_every = 0
    elif params['verbose'] is True:
        log_every = 1
    else:
        log_every = params['verbose']

    sc, train_size, test_size, dataX, dataY, trainX, trainY, testX, testY = load_data(
        dataset_path=dataset_path,
        dataset_label=dataset_label
    )

    lstm = LSTM(
        params['num_classes'],
        params['input_size'],
        params['hidden_size'],
        params['num_layers']
    )

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=params['learning_rate'])  #  torch.optim.SGD

    # Train the model
    for epoch in range(params['num_epochs']):
        outputs = lstm(trainX)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, trainY)

        loss.backward()

        optimizer.step()

        if log_every > 0 and epoch % log_every == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


    lstm.eval()
    train_predict = lstm(dataX)

    data_predict = train_predict.data.numpy()
    dataY_plot = dataY.data.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)

    plt.axvline(x=train_size, c='r', linestyle='--', label='Train/Test split')

    plt.plot(dataY_plot, label='Dataset')
    plt.plot(data_predict, label='LSTM')
    plt.suptitle(f'Time-Series Prediction ({dataset_label})')
    plt.legend()
    plt.show()

