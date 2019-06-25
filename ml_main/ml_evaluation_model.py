####### For evaluation metrics #######
from sklearn.metrics import hamming_loss         #Hamming loss
from sklearn.metrics import zero_one_loss        #zero-one error
from sklearn.metrics import coverage_error 
from sklearn import metrics
####### JUST FOR TESTING...
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import numpy as np
from log_ import status_logging,error_logging


# SECTION 3 : Evaluation

#input : model, testX, testY
#output : results of evaluation formatted  as string 
#detail : using predY method, generate each model's score.
#for example, 
# model hamming oneerror coverage f1score precision
# KNN   0.87    0.76     0.79     0.89    0.83 
def make_evaluation_report(predY,testY):
    ############Configuration ################
    average_metric = 'micro' # {None,'micro', 'macro','weighted','samples'}
    
    ############Configuration end#############
    
    #1) Hamming loss
    hl = hamming_loss(testY,predY)
    #2) one_error
    one_error = 0.00000000000000001
    #3) Coverage
#     try:
    try:
        coverage = coverage_error(testY.toarray(),predY.toarray())
    except:
        try:
            coverage = coverage_error(testY,predY.todense())
        except :
            coverage =0.0
#     except:
#         coverage = coverage_error(testY,predY.todense())
    #4) F1-score and precision
    f1score = metrics.f1_score(testY,predY,average=average_metric)
    precision = metrics.precision_score(testY,predY,average=average_metric)

    res = [hl,one_error,coverage,f1score,precision]
#     print(" %-12s %-12s %-12s %-12s %-12s %-12s" % ('modelname','hamming','one_error','coverage','f1score','precision'))
    
#     print(" %-12s %-12f %-12f %-12f %-12f %-12f" % ('modelname',res[0],res[1],res[2],res[3],res[4]))
    return res
    
#just wrapper function for making report chart.

def print_report(results,modelname,feature_list):
    names = ['modelname','hamming','one_error','coverage','f1score','precision']
    basic_names = "{0:^20.17s}"
    basic= "{i}:^12.10{type_}"
    tamplete = ""
    def type_to_string(x):
        if type(x)==type('str'):
            return 's'
        else:
            return 'f'
    tablenames =""
    tablenames+=basic_names
    for i in range(5):
        tablenames += '{'+basic.format(i=i+1,type_='s')+'} '
    
    basic_names = "{0:<20.17s}"
    basic= "{i}:<12.10{type_}"
    tamplete+=basic_names
    for i in range(5):
        tamplete += '{'+basic.format(i=i+1,type_=type_to_string(results[i]))+'} '

    rows = [tablenames.format(*names),tamplete.format(modelname,*results)]
    for row in rows:
        print(row)
    with open('ml/ml_ouptut_'+feature_list+'.txt','w') as f:
        for row in rows:
            f.write(row+'\n')
    return

