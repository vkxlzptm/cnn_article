from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from generate_feature_fileio import load_feature
import numpy as np
from log_ import status_logging, error_logging
from sklearn.externals import joblib

filename="ml/"
models=['cnn']
def eval_main(file_ext,models,loaded_model_names,testX,testY):
    from ml_evaluation_model import make_evaluation_report, print_report
    results = []
    i = 0
    mn = []
    for model in models:
        filename = loaded_model_names[i] + '_'+file_ext
        try:
            status_logging('evaluate {filename} start'.format(filename = filename))
            results.append(make_evaluation_report(model,testX,testY))
            mn.append(loaded_model_names[i])
        except:
            error_logging()
            status_logging('evaluate {filename} made error'.format(filename=filename))
        i+=1
    print_report(results,mn,file_ext)
    return 
#test/train data setting




train_tag=load_feature(DATA_NAME='total',METHOD='tag',DATATYPE='train',DETAIL='train')
test_tag=load_feature(DATA_NAME='total',METHOD='tag',DATATYPE='test',DETAIL='test')
'''
print(tfidf_train_body.shape)
print(tfidf_train_title.shape)
print(tfidf_test_body.shape)
print(tfidf_test_title.shape)
print(tfidf_test_body[1,:,:])


print(train_tag.shape)
print(test_tag.shape)
'''
'''
result: 

(3422, 300, 1437)
(3422, 15, 1437)
(1467, 300, 1437)
(1467, 15, 1437)
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
(3422, 5)
(1467, 5)
'''

Xtest=joblib.load("ml/Xtest.sav")
k=list(Xtest.shape)
k.append(1)
Xtest=Xtest.reshape(k)
Ytest=joblib.load("ml/Ytest.sav")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard

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

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
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


preds = model.predict(Xtest)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

#storing


from ml_evaluation_model import print_report, make_evaluation_report


joblib.dump(preds,filename+'{}.sav'.format('Ypred'))
model.save(filename+"cnn_char.h5")
print_report(make_evaluation_report(model,Xtest,Ytest),['cnn'],'cnn')


