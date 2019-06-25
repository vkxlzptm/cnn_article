import numpy as np
from ml_evaluation_model import print_report, make_evaluation_report

#simple_pred
# 확률이 0.5만 넘으면 해당 태그에 분류
def simple_pred(predY):
    predY[predY>=0.5] = 1
    predY[predY<0.5] = 0
    return predY

#complex_pred
# 태그별 확률을 순서대로 나열하여, 가장 갭이 큰 부분을 기준으로 태그를 분류
def complex_pred(predY):
    result=[]
    for rank in predY:
        sorted_rank_arg = np.argsort(-rank)
        diffs = -np.diff([rank[k] for k in sorted_rank_arg])

        indcutt = np.where(diffs == diffs.max())[0]
        if len(indcutt.shape) == 1:
            indcut = indcutt[0] + 1
        else:
            indcut = indcutt[0, -1] + 1
        label = np.zeros(rank.shape)

        label[sorted_rank_arg[0:indcut]] = 1

        result.append(label)
    res= np.asarray(result)
    
    return res


from keras.models import load_model
from sklearn.externals import joblib

#main
#학습된 model을 가져와서 
def main():
    Xtest=joblib.load('ml/Xtest.sav')
    k=list(Xtest.shape)
    k.append(1)
    Xtest=Xtest.reshape(k)
    Ytest=joblib.load('ml/Ytest.sav')

    model=load_model('ml/cnn_char.h5')
    predY=model.predict(Xtest)

    complex_rst=complex_pred(predY)
    simple_rst=simple_pred(predY)
    print_report(make_evaluation_report(complex_rst,Ytest),'complex_cnn','complex')
    print_report(make_evaluation_report(simple_rst,Ytest),'simple_cnn','simple')

if __name__ == "__main__":
    main()