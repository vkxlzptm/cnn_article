from fileio import load_feature
import numpy as np
from sklearn.externals import joblib


test_tag=load_feature(DATA_NAME='total',METHOD='tag',DATATYPE='test',DETAIL='test')
joblib.dump(test_tag,'ml/Ytest.sav')

ohencoding_list_body=load_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='test',DETAIL='test_body')
ohencoding_list_title=load_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='test',DETAIL='test_title')

Xtest=np.concatenate((ohencoding_list_title,ohencoding_list_body),axis=1)
'''
for i in range(ohencoding_list_body.shape[0]):
    X_merge=np.concatenate((ohencoding_list_title[i],ohencoding_list_body[i]))
    Xtest.append(X_merge)
print("procedure 2 done")
Xtest=np.asarray(Xtest)
'''
print(Xtest.shape)
joblib.dump(Xtest,'ml/Xtest.sav')
print("done")
