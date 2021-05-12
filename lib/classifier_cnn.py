import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class CNN:
    def __init__(self, BN=False, verbose=False):
        # self.lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.001, decay_rate=0.5, decay_steps=1.0)
        self.model = models.Sequential()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.01)
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.initializer = initializers.HeNormal()
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None
        self.target_mean = 0
        self.target_std = 0
        self.verbose = verbose
        self.BN = BN
        self.w = 0
        self.CLASSES = 2
        self.NUM_FEATURES = 7
        self.PLOT_MODEL_FILEPATH = '../images/CNN.png'
        self.DEFAULT_MODEL_FILEPATH = '../model/CNN_weights'
        self.DEFAULT_FILEPATH = '../parsed_data/1000games_batchQ.csv.gz'

    def init_model(self):
        # conv2D :: n_filter=400, kernel=(4, 4)
        self.model.add(layers.Conv2D(filters=512, kernel_size=(4, 4), input_shape=(8, 8, 7), activation='relu', padding='same'))
        # self.model.add(layers.BatchNormalization()) if self.BN else None
        # MaxPool2D :: kernel=(2, 2), stride=(2, 2)
        self.model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        # conv2D :: n_filter=200, kernel=(2, 2)
        self.model.add(layers.Conv2D(filters=258, kernel_size=(2, 2), activation='relu', padding='same'))
        # self.model.add(layers.BatchNormalization(axis=3)) if self.BN else None
        # Flatten to single output dimension
        self.model.add(layers.Flatten())
        # LinearIP :: output_dim=70, neurons=RELU
        self.model.add(layers.Dense(units=70, activation='relu', kernel_initializer=self.initializer))
        # self.model.add(layers.BatchNormalization()) if self.BN else None
        # Dropout :: p=0.2
        self.model.add(layers.Dropout(rate=0.3))
        # LinearIP :: output_dim=1, neurons=RELU
        self.model.add(layers.Dense(units=2, activation='softmax'))
        if self.verbose:
            print('<|\tInitializing the CNN model')

    def save_model(self):
        self.model.save_weights(self.DEFAULT_MODEL_FILEPATH)

    def load_model(self):
        self.init_model()
        self.model.load_weights(self.DEFAULT_MODEL_FILEPATH)

    def model_summary(self):
        self.model.summary()
        exit()

    def plot_diff(self, y):
        kvot = (self.w/y.shape[0])
        print(f'baseline: {kvot*100}%')
        labels = ['black', 'white']
        plt.bar(labels, [1 - kvot, kvot], color=['black', 'bisque'])
        plt.ylabel('Ratio')
        plt.title('Data distribution of positions')
        plt.show()

    def normalize_labels(self, labels):
        # labels shape (572386, 1)
        new_labels = np.zeros((labels.shape[0], self.CLASSES))
        for row in range(labels.shape[0]):
            if labels[row, 0] < 0:
                new_labels[row, 0] = 1
            elif labels[row, 0] >= 0:
                new_labels[row, 1] = 1
                self.w += 1
        self.plot_diff(labels)

        """
        for row in range(labels.shape[0]):
            if labels[row, 0] >= 20:
                new_labels[row, 6] = 1
            elif 10 <= labels[row, 0] < 20:
                new_labels[row, 5] = 1
            elif 2 <= labels[row, 0] < 10:
                new_labels[row, 4] = 1
            elif -2 < labels[row, 0] < 2:
                count += 1
                new_labels[row, 3] = 1
            elif -10 < labels[row, 0] <= -2:
                new_labels[row, 2] = 1
            elif -20 < labels[row, 0] <= -10:
                new_labels[row, 1] = 1
            elif labels[row, 0] <= -20:
                new_labels[row, 0] = 1
        print(f'baseline acc: {100*count/labels.shape[0]}%')
        """
        # self.target_mean = np.mean(labels)
        # self.target_std = np.std(labels)
        # labels = (labels - np.mean(labels))/np.std(labels)
        return new_labels

    def read_files(self):
        data = []
        column_list = []
        for x in range(self.NUM_FEATURES * 8 * 8):
            column_list.append(f'x{x}')
        for file in os.listdir('../parsed_data/'):
            if '1000games' in file or '2000games' in file:
                print(f'<|\tParsing data from filepath :: ../parsed_data/{file}')
                data.append(pd.read_csv('../parsed_data/'+file))
        train_x = []
        train_y = []
        for dat in data:
            train_x.append(dat.loc[:, column_list])
            train_y.append(dat.loc[:, dat.columns == 'y'])
        return train_x, train_y

    def parse_data(self):
        x_data, y_data = self.read_files()
        x_train = np.concatenate(x_data, axis=0)
        y_train = np.concatenate(y_data, axis=0)
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices, :]
        y_train = y_train[indices, :]
        y_train = self.normalize_labels(y_train)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5)
        self.x_train = x_train.reshape(len(x_train), 8, 8, 7)
        self.x_validation = x_valid.reshape(len(x_valid), 8, 8, 7)
        self.x_test = x_test.reshape(len(x_test), 8, 8, 7)
        self.y_train = y_train
        self.y_validation = y_valid
        self.y_test = y_test
        if self.verbose:
            print(f'<|\t\tNumber of training samples :: {len(self.x_train)}')
            print(f'<|\t\tNumber of training samples :: {len(self.x_validation)}')
            print(f'<|\t\tNumber of training samples :: {len(self.x_test)}')

    def batch_train(self, n_epochs=10):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, validation_data=(self.x_validation, self.y_validation), callbacks=[callback], batch_size=128)

    def plot_history(self, hist_type='loss', xlabel='iteration', ylabel='Cross Entropy'):
        plt.plot(self.history.history[hist_type], label=hist_type)
        plt.plot(self.history.history[f'val_{hist_type}'], label=f'val_{hist_type}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def plot_histogram(self):
        plt.hist(self.y_test, bins=500)
        plt.xlabel('evaluation')
        plt.ylabel('num labels')
        plt.show()

    def plot_model(self, filepath=None):
        if filepath is None:
            filepath = self.PLOT_MODEL_FILEPATH
        tf.keras.utils.plot_model(self.model, to_file=filepath, show_shapes=True, rankdir='LR')

    def model_predict(self):
        y = self.model.predict(self.x_test)

        assert len(y) == len(self.y_test), \
            print(f'<|\t\tERROR: '
                  f'predictions and target note same length.'
                  f'\n\t\tlen(prediction)={len(y)} :: len(target)={len(self.y_test)}')
        acc = 0
        diff = []
        for (target, predicted) in zip(self.y_test, y):
            print(f'target={target} :: predicted={predicted}')
            if np.argmax(target) == np.argmax(predicted):
                acc += 1
            diff.append(np.abs(target - predicted))
        print(f'<|\tModel testing accuracy:\t {100*round(float(acc)/float(len(y)), 4)}%')


def main():
    # ----- Unit testing -----

    model = CNN(verbose=True)
    model.init_model()
    # model.load_model()
    model.parse_data()
    # model.plot_histogram()
    # model.plot_model()
    # model.model_summary()
    #"""
    model.batch_train(n_epochs=25)
    # model.save_model()
    model.plot_history(hist_type='loss')
    model.plot_history(hist_type='accuracy')
    model.model_predict()
    #"""


if __name__ == '__main__':
    main()
