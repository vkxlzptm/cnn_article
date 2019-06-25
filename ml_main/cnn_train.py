from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from fileio import load_feature
import numpy as np
from log_ import status_logging, error_logging
from sklearn.externals import joblib

Xtest=joblib.load("ml/Xtest.sav")
k=list(Xtest.shape)
k.append(1)
Xtest=Xtest.reshape(k)
Ytest=joblib.load("ml/Ytest.sav")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from os.path import isfile

if isfile('ml/cnn_char.h5'):
    from keras.models import load_model
    model=load_model('ml/cnn_char.h5')
else:
    model = Sequential()
    model.add(Conv2D(32, (3,Xtest.shape[2]),input_shape=Xtest.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(30,1)))
    model.add(Flatten()) 
    model.add(Dense(5000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(Ytest.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop')

for _ in range(3):
    for i in range(3):
        Xtrain=joblib.load("ml/Xtrain_{}.sav".format(i))
        k=list(Xtrain.shape)
        k.append(1)
        Xtrain=Xtrain.reshape(k)
        Ytrain=joblib.load("ml/Ytrain_{}.sav".format(i))
        model.fit(Xtrain, Ytrain, epochs=20, batch_size=500,  callbacks = [TensorBoard("ml/run_a")])
        del Xtrain, Ytrain

#storing model

model.save("ml/cnn_char.h5")


