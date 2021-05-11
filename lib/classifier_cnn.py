import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models,initializers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, verbose=False):
        self.lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.001, decay_rate=0.5, decay_steps=1.0)
        self.model = models.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.BinaryCrossentropy()
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
        self.NUM_FEATURES = 7
        self.PLOT_MODEL_FILEPATH = '../images/CNN.png'
        self.DEFAULT_MODEL_FILEPATH = '../model/CNN_weights'
        self.DEFAULT_FILEPATH = '../parsed_data/1000games_batchQ.csv.gz'

    def init_model(self):
        # conv2D :: n_filter=400, kernel=(4, 4)
        self.model.add(layers.Conv2D(filters=400, kernel_size=(4, 4), input_shape=(8, 8, 7)))
        # MaxPool2D :: kernel=(2, 2), stride=(2, 2)
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
        # conv2D :: n_filter=200, kernel=(2, 2)
        self.model.add(layers.Conv2D(filters=200, kernel_size=(2, 2)))
        # Flatten to single output dimension
        self.model.add(layers.Flatten())
        # LinearIP :: output_dim=70, neurons=RELU
        self.model.add(layers.Dense(units=70, activation='relu'))
        # Dropout :: p=0.2
        self.model.add(layers.Dropout(rate=0.3))
        # LinearIP :: output_dim=1, neurons=RELU
        self.model.add(layers.Dense(units=1, activation='sigmoid'))
        if self.verbose:
            print('<|\tInitializing the CNN model')

    def save_model(self):
        self.model.save_weights(self.DEFAULT_MODEL_FILEPATH)

    def load_model(self):
        self.init_model()
        self.model.load_weights(self.DEFAULT_MODEL_FILEPATH)

    def model_summary(self):
        self.model.summary()

    def plot_diff(self, w, b):
        print(f'baseline: {(w/(w+b))*100}%')
        labels = ['black win', 'white win']
        plt.bar(labels, [b, w])
        plt.legend()
        plt.show()

    def normalize_labels(self, labels):
        labels[labels >= 0] = 1
        labels[labels < 0] = 0
        num_white = 0
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 1:
                    num_white += 1

        num_black = len(labels) - num_white
        self.plot_diff(num_white, num_black)
        # self.target_mean = np.mean(labels)
        # self.target_std = np.std(labels)
        # labels = (labels - np.mean(labels))/np.std(labels)
        return labels

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

    def parse_data(self, filepath=None):
        if filepath is None:
            filepath = self.DEFAULT_FILEPATH
        x_data, y_data = self.read_files()
        x_train = np.concatenate(x_data, axis=0)
        y_train = np.concatenate(y_data, axis=0)
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
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, validation_data=(self.x_validation, self.y_validation), callbacks=[callback])

    def plot_history(self, hist_type='loss', xlabel='epoch', ylabel='loss'):
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

            if target == 0 and predicted < 0.5:
                acc += 1
            if target == 1 and predicted > 0.5:
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
    #model.plot_model()
    #"""
    model.batch_train(n_epochs=1)
    # model.save_model()
    model.plot_history()
    model.model_predict()
    #"""
    # Do classification, dataproblems? how can we solve it => regression


if __name__ == '__main__':
    main()
