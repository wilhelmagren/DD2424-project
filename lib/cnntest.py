import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout2d, functional, MSELoss
from torch.optim import Adam, SGD
dev = "cpu"

torch.manual_seed(69)

train = pd.DataFrame(pd.read_csv('../parsed_data/parsed_games_test.csv'))
x_train = train.loc[:, train.columns != 'y']
y_train = train.loc[:, train.columns == 'y']

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5)

# print(f'({x_train.shape}, {y_train.shape})   ...   ({x_valid.shape}, {y_valid.shape})')

x_train = x_train.reshape(len(x_train), 7, 8, 8)
x_train = torch.tensor(x_train, dtype=torch.long)

y_train = y_train.astype(float)
y_train = torch.tensor(y_train, dtype=torch.float)
#---------Validation--------------------#
x_valid = x_valid.reshape(len(x_valid), 7, 8, 8)
x_valid = torch.tensor(x_valid, dtype=torch.long)

y_valid = y_valid.astype(float)
y_valid = torch.tensor(y_valid, dtype=torch.float)

#---------Test----------------------------#
x_test = x_test.reshape(len(x_test), 7, 8, 8)
x_test = torch.tensor(x_test, dtype=torch.long)

y_test = y_test.astype(float)
y_test = torch.tensor(y_test, dtype=torch.float)

x_train, x_valid, x_test = Variable(x_train), Variable(x_valid), Variable(x_test)
y_train, y_valid, y_test = Variable(y_train), Variable(y_valid), Variable(y_test)


class Net(Module):
    def __init__(self, in_channels=1):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=400, kernel_size=(4, 4), padding=0)
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=400, out_channels=200, kernel_size=(2, 2), padding=0)
        self.fc1 = Linear(in_features=200, out_features=70)
        self.fc2 = Linear(in_features=70, out_features=1)
        self.drop = Dropout2d(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = functional.relu(self.fc1(x))
        x = self.drop(x)
        x = functional.relu(self.fc2(x))
        return x


cnn_model = Net(in_channels=7)

optimizer = Adam(cnn_model.parameters(), lr=0.05)
loss_func = MSELoss()
#---------------Training----------------------#
train_losses = []
val_losses = []
epochs = 3

for i in tqdm(range(epochs)):

    cnn_model.train()
    tr_loss = 0
    optimizer.zero_grad()

    output_train = cnn_model(x_train)
    output_val = cnn_model(x_valid)
    # computing the training and validation loss
    loss_train = loss_func(output_train, y_train)
    loss_val = loss_func(output_val, y_valid)

    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if i%2 == 0:
        # printing the validation loss
        print('Epoch : ',i+1, '\t', 'loss :', loss_val)

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

#Model:



