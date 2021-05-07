import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models,initializers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, verbose=False):
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
        self.target_mean = 0
        self.target_std = 0
        self.verbose = verbose
        self.NUM_FEATURES = 7
        self.DEFAULT_MODEL_FILEPATH = '../model/CNN_weights'
        self.DEFAULT_FILEPATH = '../parsed_data/1000games_batchQ.csv.gz'

    def init_model(self):
        """
        # conv2D :: n_filter=400, kernel=(4, 4)
        self.model.add(layers.Conv2D(filters=400, kernel_size=(4, 4), input_shape=(8, 8, 7)))
        # MaxPool2D :: kernel=(2, 2), stride=(2, 2)
        self.model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # conv2D :: n_filter=200, kernel=(2, 2)
        self.model.add(layers.Conv2D(filters=200, kernel_size=(2, 2)))
        # Flatten to single output dimension
        self.model.add(layers.Flatten())
        # LinearIP :: output_dim=70, neurons=RELU
        self.model.add(layers.Dense(units=70, activation='relu'))
        # Dropout :: p=0.2
        self.model.add(layers.Dropout(rate=0.3))
        # LinearIP :: output_dim=1, neurons=RELU
        self.model.add(layers.Dense(units=1))
         """
        self.model.add(layers.Conv2D(filters=32, kernel_size = (3,3),input_shape = (8,8,7),activation='relu',kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.AveragePooling2D(pool_size = (2,2),strides=(1,1)))
        self.model.add(layers.Dropout(rate=0.3))
        self.model.add(layers.Conv2D(filters=64,kernel_size = (3,3),activation='relu',kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.AveragePooling2D(pool_size = (2,2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256,activation ='relu',kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.Dropout(rate=0.3))
        self.model.add(layers.Dense(1,activation='linear',kernel_initializer=initializers.HeUniform()))
        if self.verbose:
            print('<|\tInitializing the CNN model')

    def save_model(self):
        self.model.save_weights(self.DEFAULT_MODEL_FILEPATH)

    def load_model(self):
        self.init_model()
        self.model.load_weights(self.DEFAULT_MODEL_FILEPATH)

    def model_summary(self):
        self.model.summary()

    def normalize_labels(self, labels):
        labels[labels > 80] = 80
        labels[labels < -80] = -80
        self.target_mean = np.mean(labels)
        self.target_std = np.std(labels)
        labels = (labels - np.mean(labels))/np.std(labels)
        return labels

    def read_files(self):
        data = []
        column_list = []
        for x in range(self.NUM_FEATURES * 8 * 8):
            column_list.append(f'x{x}')
        for idx in range(1, 7):
            print(f'<|\tParsing the data from filepath :: {self.DEFAULT_FILEPATH.replace("Q", f"{idx}")}')
            data.append(pd.read_csv(self.DEFAULT_FILEPATH.replace('Q', f'{idx}')))
        data.append(pd.read_csv('../parsed_data/2000games_batch7.csv.gz'))
        train_x = []
        train_y = []
        for dat in data:
            train_x.append(dat.loc[:, column_list])
            train_y.append(dat.loc[:, dat.columns == 'y'])
        return train_x, train_y

    def parse_data(self, filepath=None, compression='gzip'):
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

    def batch_train(self, optimizer=None, loss=None, n_epochs=10):
        if self.verbose:
            print(f'<|\tTraining the model utilizing big batch')
            if optimizer is None:
                print(f'<|\t\tNo optimizer specified\t\t=>  using default: {self.optimizer}')
                optimizer = self.optimizer
            if loss is None:
                print(f'<|\t\tNo loss specified\t\t\t=>  using default: {self.loss}')
            loss = self.loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.compile(optimizer=optimizer, loss=loss)
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

    def model_predict(self, offset=0.5):
        y = self.model.predict(self.x_test)

        assert len(y) == len(self.y_test), \
            print(f'<|\t\tERROR: '
                  f'predictions and target note same length.'
                  f'\n\t\tlen(prediction)={len(y)} :: len(target)={len(self.y_test)}')
        acc = 0
        diff = []
        diff_vanilla = []
        for (target, predicted) in zip(self.y_test, y):
            #print(f'target={target} :: predicted={predicted}')
            #if target - offset <= predicted <= target + offset:
            #    acc += 1
            target_vanilla = (target + self.target_mean)*self.target_std
            predicted_vanilla = (predicted + self.target_mean)*self.target_std
            diff.append(np.abs(target - predicted))
            diff_vanilla.append(np.abs(target_vanilla - predicted_vanilla))
            print(f'\t{predicted}  =>  {predicted_vanilla}  vs  {target_vanilla}')
        # print(f'<|\tModel testing accuracy:\t {100*round(float(acc)/float(len(y)), 4)}%')
        print(f'<|\tModel mean error normalized:\t\t {np.mean(np.array(diff))}')
        print(f'<|\tModel mean error:\t\t\t {np.mean(np.array(diff_vanilla))}')


def main():
    # ----- Unit testing -----

    model = CNN(verbose=True)
    model.init_model()
    # model.load_model()
    model.parse_data()
    # model.plot_histogram()
    #"""
    model.batch_train(n_epochs=40)
    model.save_model()
    model.plot_history()
    model.model_predict()
    #"""


if __name__ == '__main__':
    main()
