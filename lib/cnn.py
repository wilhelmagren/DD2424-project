"""
+|++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 | Project work for group 70 in the course DD2424, Deep Learning in Data Science.
 | This file implements a Convolutional Neural Network (CNN) with tensorflow.keras.
 | The CNN is used for static evaluation of chess positions. Features loading, saving, visualization, and summary of
 | the defined model. Uses homebrewn data do train and test. The data comes from publicly available games at the
 | FICS Games Database and is modeled as (8x8x7). Where each position on the chess board is represented by a feature
 | vector of length 7. That is, the channel depth of the CNN is 7.
 |
 | The final modeled is of the form [conv-relu]-[affine-elu-dropout-affine-elu]-linear, and achieved a
 | Mean Absolute Error (MAE) of 4.03 after training for 40 epochs.
 |
 | Authors: Eric Bröndum, Christoffer Torgilsman, Wilhelm Ågren
 | Last edited: 19/05-2021
+|++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class CNN:
    """
    CNN class implemented using tensorflow.keras, and handles the entire workflow of reading data,
    training, and evaluating.

    func __init__/5
    @spec :: (float, float, boolean, boolean) => Class(CNN)
        Sequential keras model, takes the hyperparameters eta and delta.
        Specifies which optimizer to use for training in callable arg.
        Verbose specifies whether or not to use prints in methods.

                                    LEGACY-KÅD models below
    MK I :
        self.model.add(layers.Conv2D(filters=512, kernel_size=(4, 4), input_shape=(8, 8, 7), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization()) if self.BN else None
        self.model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization(axis=3)) if self.BN else None
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=70, activation='relu', kernel_initializer=self.initializer))
        self.model.add(layers.BatchNormalization()) if self.BN else None
        self.model.add(layers.Dropout(rate=0.3))
        self.model.add(layers.Dense(units=1, activation='linear', kernel_initializer=self.initializer))

    MK II :
        self.model.add(layers.Conv2D(filters=128, kernel_size=(4, 4), input_shape=(8, 8, 7), activation='relu', kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(70, activation='relu', kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.Dropout(rate=0.3))
        self.model.add(layers.Dense(1, activation='linear', kernel_initializer=initializers.HeUniform()))
    """
    def __init__(self, learning_rate, delta, normalize, verbose=False) -> None:
        self.model = models.Sequential()
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        self.loss = tf.keras.losses.Huber(delta=delta)
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
        self.x_mean = 0
        self.x_std = 0
        self.studentized_residual = normalize
        self.verbose = verbose
        self.NUM_FEATURES = 7
        self.PLOT_MODEL_FILEPATH = '../images/CNN.png'
        self.DEFAULT_MODEL_FILEPATH = '../model/CNN_weights'
        self.DEFAULT_FILEPATH = '../parsed_data/1000games_batchQ.csv.gz'

    def init_model(self) -> None:
        """
        func init_model/1
        @spec (Class(CNN)) => None
            Adds all of the corresponding layers to the sequential model defined in __init__/5.
            Simply modifies the class attribute self.model
        """
        self.model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=(8, 8, 7), activation='relu', kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='elu', kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.Dropout(rate=0.3))
        self.model.add(layers.Dense(128, activation='elu', kernel_initializer=initializers.HeUniform()))
        self.model.add(layers.Dense(1, activation='linear', kernel_initializer=initializers.HeUniform()))
        if self.verbose:
            print('<|\tInitializing the CNN model')

    def save_model(self) -> None:
        """
        Save the model weights to file self.DEFAULT_MODEL_FILEPATH.
        Requires the model to have been initialized, and trained for at
        least 1 epoch such that the weights are not None.
        Once again only modifies the class attribute self.model
        """
        self.model.save_weights(self.DEFAULT_MODEL_FILEPATH)

    def load_model(self) -> None:
        """
        Load the saved model from file self.DEFAULT_MODEL_FILEPATH.
        Requires that the model is initialized prior to loading, because you can't load
        the saved weights to an empty model!
        """
        self.init_model()
        self.model.load_weights(self.DEFAULT_MODEL_FILEPATH)

    def model_summary(self) -> None:
        """
        Prints the summary of the defined model. Requires the model to have been initialized
        prior to calling. Shows the layers, how many trainable params are in each, and the total
        amount of trainable parameters.
        """
        self.model.summary()

    def normalize_labels(self, labels) -> np.array:
        """
        func normalize_labels/2
        @spec :: (Class(CNN), np.array) => np.array
            Shrink the extreme outliers and limit them to -60 and 60.
            Normalize them according to the Studentized residual,
            [https://en.wikipedia.org/wiki/Normalization_(statistics)], i.e. x = (x - mean(x))/variance(x).
            Stores the label mean and variance in class attributes, and returns the normalized labels.
        """
        labels[labels > 80] = 60
        labels[labels < -80] = -60
        if self.studentized_residual:
            self.target_mean = np.mean(labels)
            self.target_std = np.std(labels)
            labels = (labels - np.mean(labels))/np.std(labels)

        return labels

    def normalize_data(self, x) -> np.array:
        """
        func normalize_data/2
        @spec :: (Class(CNN), np.array) => np.array
            Normalize the data according to Studentized residual,
            similarly as func normalize_labels/2. Not sure if this is a good thing to do?
            Then we are not representing the same board anymore, and the problem is not the
            absolute sizes of the datapoints, but instead the amount of targets centered around 0.
            ONLY USE THIS FOR TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        return (x - self.x_mean)/self.x_std

    def read_files(self) -> (list, list):
        """
        func read_files/1
        @spec :: (Class(CNN)) => (list, list)
            Read all of the parsed_data found in '../parsed_data/' and extract
            the targets and data respectively. Returns the a list of the data and
            a list of the corresponding targets. Order of the elements are important! Can't be changed!!!
        """
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
        """
        Parse the data, ye I don't know really. This is a mess...
        """
        x_data, y_data = self.read_files()
        x_train = np.concatenate(x_data, axis=0)
        y_train = np.concatenate(y_data, axis=0)
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices, :]
        y_train = y_train[indices, :]
        y_train = self.normalize_labels(y_train)
        # x_train = self.normalize_data(x_train)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3)
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

    def plot_history(self, hist_type='loss', xlabel='epochs', ylabel='Huber loss'):
        plt.plot(self.history.history[hist_type], label=f'training {hist_type}', linewidth=1, color='maroon')
        plt.plot(self.history.history[f'val_{hist_type}'], label=f'validation {hist_type}', linewidth=1, color='navy')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def plot_histogram(self):
        plt.hist(self.y_test, bins=160, color='maroon')
        plt.xlabel('target evaluation')
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

        diff_MAE, diff_MSE = [], []
        for (target, predicted) in zip(self.y_test, y):
            if self.studentized_residual:
                vanilla_target = target*self.target_std + self.target_mean
                vanilla_prediction = predicted*self.target_std + self.target_mean
                print(f'real_target={round(vanilla_target[0]/1, 3)}\t::\treal_predicted={round(vanilla_prediction[0]/1, 3)}')
            print(f'target={round(target[0]/1, 3)}\t\t::\t\tpredicted={round(predicted[0]/1, 3)}\n')
            diff_MAE.append(np.abs(target - predicted))
            diff_MSE.append(np.square(target - predicted))
        # print(f'<|\tModel testing accuracy:\t {100*round(float(acc)/float(len(y)), 4)}%')
        print(f'<|\tModel mean absolute error:\t\t {np.mean(np.array(diff_MAE))}')
        print(f'<|\tModel mean square error:\t\t {np.mean(np.array(diff_MSE))}')


def main():
    # ----- Unit testing -----
    model = CNN(learning_rate=0.05, delta=0.5, normalize=True, verbose=True)
    model.init_model()
    model.model_summary()
    # model.load_model()
    model.parse_data()
    model.plot_histogram()
    model.plot_model()
    model.batch_train(n_epochs=40)
    model.save_model()
    model.plot_history()
    model.model_predict()


if __name__ == '__main__':
    main()
