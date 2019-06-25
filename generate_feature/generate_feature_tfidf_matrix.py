from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import _pickle as cPickle
import pandas as pd 
import json
from log_ import status_logging
from generate_feature_fileio import *

###########------OVERVIEW------###########
"""
This code generates making feature file for ml algorithm from raw STT data.
The main function consists of two part.
1) read_files : read raw STT data and return
raw STT data means dataframe with three label; 
    - 'id' : identifier for indexing
    - 'custcnselcd' : tag infomation
    - 'ratogtxt' : STT data
2) generate_{tfidf/tfidf/embedding}_{train/test} : genereate tfidf/tfidf/embedding feature file for train/test.
total 3*2 =6 fucntion is defined as below.

options : only one option is available; 'splitmethod'
each 'generate_feature' fuction is given input named 'splitmethod', just for making name of output file.
main function controls the splitmethod, and the data is splitted by type of splitmethod.  type of splitmethod is below.
    - 'naiverandom' : literally naive random. randomly shuffle input file and split them with train-test 7:3
    - 'tagrandom' : shuffle input file and split them with ratio 7:3 by tags



"""
###########----OVERVIEW END----###########


#GLOVAL VARIABLE SETTING

# here is gloval variable for this code. 

# For making filename,string formats.
#                 "feature/{tfidf|count}/{train|test}/{naiverandom|tagrandom|sampling}_train/voca/test.pkl"    






####TODO : NEED TO MODIFIED.

    

# 2) genearte_feature_train/test functions
# this kind of functions require 'text_input' and 'splitmethod'.
# text_input is from read_files(), splitmethod is just for file naming; see the OVERVIEW
# following six functions (from (2) ~ (7)) will omit the explanation of these two common inputs. 

#(2) generate tfidf train
# input: text_input,splitmethod
# output: None
# generate tfidf feature file for train with given splited text_input. Note that split process is done in main function

# detail:
# TFIDF is accronym of term-frequency / inverse document-frequency. (please reference WIKI)
# this fuction generates sparse matrix of tfidf vectors, for given input.
# if M STT data is given, then matrix size would be (M,dim). Here dim is determined by vectorizer objects, with input like max_df,min_df.
# [max_df] to ignore the word with the document frequency rate higher that 'max_df'; [min_df] to ignore the word with lower document frequency lower thas 'min_df' 
###TFIDF
def generate_tfidf_train(text_input):
    #1: create vectorizer obj
    tfidf_vectorizer_train = TfidfVectorizer(analyzer="char")
    #2: generate tfidf vectors using given STT data
    status_logging("generate {name} from STT".format(name='tfidf feature for train'))
    tfidf_vectorizer_train = tfidf_vectorizer_train.fit(text_input.loc[:,['title','body']].apply(lambda x: ''.join(x),axis=1))
    vocabulary = tfidf_vectorizer_train.vocabulary_
    
    ohencoding_list_body=list()
    ohencoding_list_title=list()
    for i in range(text_input.shape[0]):
        tfidf_Train_body=tfidf_vectorizer_train.transform(list(text_input.iat[i,1])).toarray()
        tfidf_Train_title=tfidf_vectorizer_train.transform(list(text_input.iat[i,0])).toarray()
        ohencoding_list_body.append(tfidf_Train_body)
        ohencoding_list_title.append(tfidf_Train_title)
    

    '''
    tfidf_Train_body= tfidf_vectorizer_train.transform(text_input['body']).toarray()
    tfidf_Train_title= tfidf_vectorizer_train.transform(text_input['title']).toarray()
    '''
    status_logging("generated {name} from STT".format(name='tfidf feature for test'))


    save_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='train',DETAIL='voca',res=vocabulary)
    save_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='train',DETAIL='train_body',res=np.asarray(ohencoding_list_body))
    save_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='train',DETAIL='train_title',res=np.asarray(ohencoding_list_title))
    
    return

#(5) generate tfidf test
# input: text_input,splitmethod
# output: None
# generate tfidf feature file for test with given splited text_input. Note that split process is done in main function
###tfidf
def generate_tfidf_test(text_input):
    
    #1: read vocabulary
    vocabulary = load_voca(DATA_NAME='total',METHOD='tfidf',DATATYPE='train',DETAIL='voca')
    #2: generate tfidf vectors using given STT data
    tfidf_vectorizer_test = TfidfVectorizer(analyzer="char",vocabulary=vocabulary)

    status_logging("generate {name} from STT".format(name='tfidf feature for test'))

    ohencoding_list_body=list()
    ohencoding_list_title=list()
    for i in range(text_input.shape[0]):
        tfidf_Test_body=tfidf_vectorizer_test.fit_transform(list(text_input.iat[i,1])).toarray()
        tfidf_Test_title=tfidf_vectorizer_test.fit_transform(list(text_input.iat[i,0])).toarray()
        ohencoding_list_body.append(tfidf_Test_body)
        ohencoding_list_title.append(tfidf_Test_title)
    '''
    tfidf_Test_body = tfidf_vectorizer_test.fit_transform(text_input['body'])
    tfidf_Test_title = tfidf_vectorizer_test.fit_transform(text_input['title'])
    '''
    status_logging("generated {name} from STT".format(name='tfidf feature for test'))
    #3: tfidf_test saving
    #outfilename_tfidf_test = "feature/test_tfidf.pkl"
    # tfidf_Test = tfidf_vectorizer_test
    save_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='test',DETAIL='test_body',res=np.asarray(ohencoding_list_body))
    save_feature(DATA_NAME='total',METHOD='tfidf',DATATYPE='test',DETAIL='test_title',res=np.asarray(ohencoding_list_title))
    return
