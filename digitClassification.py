import time

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

def create_model():
    model = Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(397, activation='relu', input_shape=(784,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(x_train,Y_train, folds=5):
    scores, histories = list(), list()
    kfold = KFold(folds, shuffle=True, random_state=1) #cross validation
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    for train_index,test_index in kfold.split(x_train,Y_train):
        model = create_model()
        xTrain,yTrain,xTest,yTest = x_train[train_index],Y_train[train_index],x_train[test_index],Y_train[test_index] #train and test index

        history = model.fit(xTrain,yTrain,epochs=50,validation_data=(xTest,yTest),verbose=2)

        #evaluate model
        _,acc = model.evaluate(x_test,Y_test,verbose=2)
        print('> %.3f' % (acc * 100.0))

        scores.append(acc)
        histories.append(history)

    return histories


def model_mean_summary(histories):
    losses = list()
    val_losses = list()
    mean_val_values = list()
    mean_values = list()
    for i in range (len(histories)):
        losses.append(histories[i].history['loss'])
        val_losses.append(histories[i].history['val_loss'])
    print(losses)
    # mean_val_values = mean(val_losses,axis=0)
    # mean_values = mean(losses,axis=0)
    # pyplot.plot(losses,label='train loss')
    # pyplot.plot(val_losses, label='validate loss')
    # pyplot.show()



4
def run_test():
    histories = evaluate_model(x_train,Y_train)
    model_mean_summary(histories)
    # summarize_performance(scores)
    pyplot.show()



run_test()
