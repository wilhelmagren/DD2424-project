import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout2d
from torch.optim import Adam, SGD
dev = "cpu"

train = pd.DataFrame(pd.read_csv('../parsed_data/parsed_games_test.csv'))
x_train = train.loc[:,train.columns != 'y']
y_train = train.loc[:,train.columns == 'y']

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

x_valid, x_test, y_valid, y_test = train_test_split(x_valid,y_valid,test_size=0.5)

x_train = x_train.reshape(len(x_train),7,8,8)
x_train = torch.from_numpy(x_train).long()

y_train = y_train.astype(float)
y_train  = torch.from_numpy(y_train).long()
#---------Validation--------------------#
x_valid = x_valid.reshape(len(x_valid),7,8,8)
x_valid = torch.from_numpy(x_valid)

y_valid = y_valid.astype(float)
y_valid = torch.from_numpy(y_valid)

#---------Test----------------------------#
x_test = x_test.reshape(len(x_test),7,8,8)
x_test = torch.from_numpy(x_test)

y_test = y_test.astype(float)
y_test = torch.from_numpy(y_test)



class Net(Module):
    def __init__(self):
        super(Net,self).__init__()

        self.cnn_layers = Sequential(
            #First layer goes from 7*8*8*batch -> 400*5*5*batch
            Conv2d(7,400,kernel_size=(4,4),stride=1,padding=0),
            MaxPool2d(kernel_size = (2,2),stride=(2,2)),
            Conv2d(400,200,kernel_size=(2,2),stride=1),
            Linear(2*2*200,70),
            Dropout2d(p=0.2),
            Linear(70,1),
            #Elu(inPlace=True)
        )

    def forward(self,x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0),-1)
        x = self.linear_layers(x)
        return x

cnn_model = Net()

optimizer = Adam(cnn_model.parameters(),lr=0.05)
criterion = CrossEntropyLoss()
#---------------Training----------------------#
train_losses = []
val_losses = []
epochs = 25

for i in tqdm(range(epochs)):

    cnn_model.train()
    tr_loss = 0
    x_train,y_train = Variable(x_train), Variable(y_train)
    x_valid,y_valid = Variable(x_valid), Variable(y_valid)
    optimizer.zero_grad()

    output_train = cnn_model(x_train)
    output_val = cnn_model(x_valid)
    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_valid)

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



