import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# -------------------- DATA ----------------------
train = pd.DataFrame(pd.read_csv('../parsed_data/parsed_games_test.csv'))
x_train = train.loc[:, train.columns != 'y']
y_train = train.loc[:, train.columns == 'y']

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5)

x_train = x_train.reshape(len(x_train), 8, 8, 7)
x_valid = x_valid.reshape(len(x_valid), 8, 8, 7)
x_test = x_test.reshape(len(x_test), 8, 8, 7)

# -------------------- MODEL ----------------------
model = models.Sequential()
# conv2D :: n_filter=400, kernel=(4, 4)
model.add(layers.Conv2D(filters=400, kernel_size=(4, 4), input_shape=(8, 8, 7)))
# MaxPool2D :: kernel=(2, 2), stride=(2, 2)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# conv2D :: n_filter=200, kernel=(2, 2)
model.add(layers.Conv2D(filters=200, kernel_size=(2, 2)))
# Flatten to single output dimension
model.add(layers.Flatten())
# LinearIP :: output_dim=70, neurons=RELU
model.add(layers.Dense(units=70, activation='linear'))
# Dropout :: p=0.2
model.add(layers.Dropout(rate=0.2))
# LinearIP :: output_dim=1, neurons=RELU
model.add(layers.Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.MSE)
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_valid, y_valid))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
y = model.predict(x_test)
acc = 0
tot = 0
for (realy, predy) in zip(y_train, y):
    if realy - 0.5 < predy < realy + 0.5:
        acc += 1
    tot += 1
print(f' testing accuracy: {100*(float(acc)/float(tot))}%')



