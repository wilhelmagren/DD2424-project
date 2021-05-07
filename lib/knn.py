import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models,initializers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k=2, verbose=False):
        self.model = models.Sequential()
        self.optimizer = 'adam'
        self.loss = tf.keras.losses.MSE
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None
        self.verbose = verbose
        self.DEFAULT_MODEL_FILEPATH = '../model/CNN_weights'

    def init_model(self):
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))
        self.model.add(layers.Dense(1024, activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))
        self.model.add(layers.Dense(1, activation='relu'))
        if self.verbose:
            print('<|\tInitializing the KNN model')

    def save_model(self):
        self.model.save_weights(self.DEFAULT_MODEL_FILEPATH)

    def load_model(self):
        self.init_model()
        self.model.load_weights(self.DEFAULT_MODEL_FILEPATH)

    def model_summary(self):
        self.model.summary()

    def parse_data(self, filepath=None, compression='gzip'):
        if filepath is None:
            filepath = self.DEFAULT_FILEPATH
        if self.verbose:
            print(f'<|\tParsing the data from filepath :: {filepath}')
        column_list = []
        for x in range(self.NUM_FEATURES * 8 * 8):
            column_list.append(f'x{x}')
        train = pd.read_csv(filepath, compression=compression)
        train2 = pd.read_csv('../parsed_data/1000games_batch2.csv.gz', compression=compression)
        train4 = pd.read_csv('../parsed_data/1000games_batch4.csv.gz', compression=compression)
        x_train = np.concatenate((train.loc[:, column_list], train2.loc[:, column_list], train4.loc[:, column_list]), axis=0)
        y_train = np.concatenate((train.loc[:, train.columns == 'y'], train2.loc[:, train.columns == 'y'], train4.loc[:, train.columns == 'y']), axis=0)
        x_train = x_train
        y_train = y_train
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


