from sklearn.feature_extraction.text import CountVectorizer
import _pickle as cPickle
import pandas as pd
from log_ import status_logging
from generate_feature_fileio import *
#(3) generate count train
# input: text_input,splitmethod
# output: None
# generate count feature file for train with given splited text_input. Note that split process is done in main function

# detail:
# this fuction generates sparse matrix of count vectors, for given input.
# count vector means each dim represents count of occurence of according word. 
# if M STT data is given, then matrix size would be (M,dim). Here dim is determined by vectorizer objects, with input like max_df,min_df.
###COUNT

def generate_count_train(text_input):
    #1: create vectorizer obj
    count_vectorizer_train = CountVectorizer(analyzer="char")
    #2: generate count vectors using given STT data
    status_logging("generate {name} from STT".format(name='count feature for train'))
    count_Train = count_vectorizer_train.fit(text_input.loc[:,['title','body']].apply(lambda x: ''.join(x),axis=1))
    vocabulary = count_Train.vocabulary_
    count_Train_body= count_Train.transform(text_input['body']).toarray()
    count_Train_title= count_Train.transform(text_input['title']).toarray()
    status_logging("generated {name} from STT".format(name='count feature for test'))


    save_feature(DATA_NAME='total',METHOD='count',DATATYPE='train',DETAIL='voca',res=vocabulary)
    save_feature(DATA_NAME='total',METHOD='count',DATATYPE='train',DETAIL='train_body',res=count_Train_body)
    save_feature(DATA_NAME='total',METHOD='count',DATATYPE='train',DETAIL='train_title',res=count_Train_title)
    
    return

#(5) generate count test
# input: text_input,splitmethod
# output: None
# generate count feature file for test with given splited text_input. Note that split process is done in main function
###count
def generate_count_test(text_input):
    
    #1: read vocabulary
    vocabulary = load_voca(DATA_NAME='total',METHOD='count',DATATYPE='train',DETAIL='voca')
    #2: generate count vectors using given STT data
    count_vectorizer_test = CountVectorizer(analyzer="char",vocabulary=vocabulary)

    status_logging("generate {name} from STT".format(name='count feature for test'))
    count_Test_body = count_vectorizer_test.fit_transform(text_input['body'])
    count_Test_title = count_vectorizer_test.fit_transform(text_input['title'])
    status_logging("generated {name} from STT".format(name='count feature for test'))
    #3: count_test saving
    #outfilename_count_test = "feature/test_count.pkl"
    # count_Test = count_vectorizer_test
    save_feature(DATA_NAME='total',METHOD='count',DATATYPE='test',DETAIL='test_body',res=count_Test_body)
    save_feature(DATA_NAME='total',METHOD='count',DATATYPE='test',DETAIL='test_title',res=count_Test_title)
    return