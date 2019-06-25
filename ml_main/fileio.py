
from log_ import *

from sklearn import preprocessing
import pandas as pd
import json
import _pickle as cPickle
import numpy as np
from sklearn.externals import joblib

FEATURE_DATAPATH ="feature/{DATA_NAME}/{DATATYPE}/{METHOD}_{DETAIL}.sav" # for pickle file output(train/test)
MORPH_DATAPATH = "crawling/puri_news_data_{DATA_NAME}.pkl"

#(1) read_files(void) : returns raw STT data
def read_files(data_name):
    # ###FOR TEST###
    # for test, read small size data
    status_logging("load {name}".format(name=data_name))
    with open(MORPH_DATAPATH.format(DATA_NAME=data_name),'rb') as infile:
        df = joblib.load(infile)
    # ###FOR TEST END
    #FOR REAL #read all morph data
    df['title']=df['title'].map(lambda x: x[:15])
    df['body']=df['body'].map(lambda x: x[:300])

    status_logging("load end {name}".format(name=data_name))
    return df
    #FOR REAL END
    
    

def load_voca(DATA_NAME=None,METHOD=None,DATATYPE=None,DETAIL=None):
    vocafilename = FEATURE_DATAPATH.format(DATA_NAME=DATA_NAME,METHOD=METHOD,DATATYPE=DATATYPE,DETAIL=DETAIL)
    status_logging("load {name} from {save_path}".format(name=METHOD,save_path=vocafilename))
    #with open(vocafilename, "rb") as infile:
    vocab_ = joblib.load(vocafilename)
    status_logging("load end {name} from {save_path}".format(name=METHOD,save_path=vocafilename))
    return vocab_


def load_feature(DATA_NAME=None,METHOD=None,DATATYPE=None,DETAIL=None):
    inputfilename = FEATURE_DATAPATH.format(DATA_NAME=DATA_NAME,METHOD=METHOD,DATATYPE=DATATYPE,DETAIL=DETAIL)
    status_logging("load {name} from {save_path}".format(name=METHOD,save_path=inputfilename))
    #with open(inputfilename, "rb") as infile:
    res = joblib.load(inputfilename)
    print("loading end ",inputfilename,res.shape)
    status_logging("load end {name} from {save_path}".format(name=METHOD,save_path=inputfilename))
    return res 


def save_feature(DATA_NAME=None,METHOD=None,DATATYPE=None,DETAIL=None,res=None):
    outputfilename=FEATURE_DATAPATH.format(DATA_NAME=DATA_NAME,METHOD=METHOD,DATATYPE=DATATYPE,DETAIL=DETAIL)
    status_logging("saving {dataname} in {save_path}".format(dataname=METHOD,save_path=outputfilename) )
    #First step: check the existence of directory on path. If not, make directory
    import os, errno
    if not os.path.exists(os.path.dirname(outputfilename)):
        try:
            os.makedirs(os.path.dirname(outputfilename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    #Second step: write file
    #with open(outputfilename, "wb") as outfile:
    joblib.dump(res, outputfilename)
    status_logging("{dataname} saved in {save_path}".format(dataname=METHOD,save_path=outputfilename) )
    return 