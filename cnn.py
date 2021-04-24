import time

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from matplotlib import pyplot
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.debug.examples.debug_mnist import tf
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.optimizer_v1 import SGD
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import keras
import statistics
from numpy import mean
from keras.callbacks import EarlyStopping
from numpy import std

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

x_train = train.drop('label', axis=1)
y_train = train['label']
x_test = test.drop('label', axis=1)
y_test = test['label']


#NORMALIZE THE DATA

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float")/255.0



#one-hot encoding

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0],28,28,1))

def create_model(hidden_nodes,loss_function):
    folds = 5
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(hidden_nodes, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    # opt = SGD(lr=0.01, momentum=0.6)
    model.compile(optimizer='sgd', loss=loss_function, metrics=['accuracy'])

    scores, histories = list(), list()
    kfold = KFold(folds, shuffle=True, random_state=1)  # cross validation

    for train_index, test_index in kfold.split(x_train, Y_train):
        xTrain, yTrain, xTest, yTest = x_train[train_index], Y_train[train_index], x_train[test_index], Y_train[
            test_index]  # train and test index

        history = model.fit(xTrain, yTrain, epochs=20, validation_data=(xTest, yTest),verbose=2)
        histories.append(history)
        _, acc = model.evaluate(x_test, Y_test, verbose=2)
        print('> %.3f' % (acc * 100.0))

    return histories

def create_plots(all_histories):
    epochs = 20
    mean_values = []
    mean_values_val = []
    nodes = [5,397,794]
    for histories in range (len(all_histories)):
        losses = []
        val_losses = []
        for i in range(5):
            losses.append(all_histories[histories][i].history['loss'])
            val_losses.append(all_histories[histories][i].history['val_loss'])
        mean_values.append(mean(losses, axis=0))
        mean_values_val.append(mean(val_losses,axis=0))
    print(mean_values)
    print("////////////////////////")
    print(mean_values_val)
    fig = pyplot.figure()
    plots = fig.add_subplot(1, 1, 1)
    plots.plot([x for x in range(1, epochs + 1)], mean_values[0], label='categorical_crossentropy')
    plots.plot([x for x in range(1, epochs + 1)], mean_values_val[0], label='categorical_crossentropy_validation')
    plots.plot([x for x in range(1, epochs + 1)], mean_values[1], label='mse')
    plots.plot([x for x in range(1, epochs + 1)], mean_values_val[1], label='mse validation')
    plots.set_xlabel("epochs")
    plots.set_ylabel("mean of loss per epoch")
    plots.legend()
    pyplot.show()




def create_plots_accuracy(all_histories):
    epochs = 20
    mean_values = []
    mean_values_val = []
    nodes = [5, 397, 794]
    for histories in range(len(all_histories)):
        accuracy = []
        val_accuracy = []
        for i in range(5):
            accuracy.append(all_histories[histories][i].history['accuracy'])
            val_accuracy.append(all_histories[histories][i].history['val_accuracy'])
        mean_values.append(mean(accuracy, axis=0))
        mean_values_val.append(mean(val_accuracy, axis=0))
    print(mean_values)
    print("////////////////////////")
    print(mean_values_val)
    fig = pyplot.figure()
    plots = fig.add_subplot(1, 1, 1)
    plots.plot([x for x in range(1, epochs + 1)], mean_values[0], label='categorical_crossentropy')
    plots.plot([x for x in range(1, epochs + 1)], mean_values_val[0], label='categorical_crossentropy_validation')
    plots.plot([x for x in range(1, epochs + 1)], mean_values[1], label='mse')
    plots.plot([x for x in range(1, epochs + 1)], mean_values_val[1], label='mse validation')
    plots.set_xlabel("epochs")
    plots.set_ylabel("mean of accuracy per epoch")
    plots.legend()
    pyplot.show()




def run_test():
    hidden_nodes = [100]
    loss_metrics = ['categorical_crossentropy','mse']
    folds = 5
    all_histories = list()
    for node in hidden_nodes:
        for loss in loss_metrics:
            all_histories.append(create_model(node,loss))
    create_plots(all_histories)
    create_plots_accuracy(all_histories)


run_test()



def averageOfNestedListsDiffEpochs(losses_nest):
    mean_values = []
    max = 0
    for list in losses_nest:
        if len(list) > max:
            max = len(list)
    for i in range(max):
        temp = []
        for list in losses_nest:
            if i < len(list):
                temp.append(list[i])
        mean_values.append(np.nanmean(temp))
    return mean_values

