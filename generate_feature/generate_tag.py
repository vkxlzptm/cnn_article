from sklearn.preprocessing import MultiLabelBinarizer
from generate_feature_fileio import load_feature,save_feature
from log_ import status_logging
from generate_feature_fileio import *

def save_tag_train(text_input):
    status_logging("tag preprocessing(train) start ")
    mlb = MultiLabelBinarizer()
    mlb_test=mlb.fit(text_input['tag'])
    tag_list=mlb_test.classes_
    tag_val=mlb.transform(text_input['tag'])
    status_logging("tag preprocessing(train) end ")

    save_feature(DATA_NAME='total',METHOD='tag',DATATYPE='train',DETAIL='voca',res=tag_list)
    save_feature(DATA_NAME='total',METHOD='tag',DATATYPE='train',DETAIL='train',res=tag_val)

    return

def save_tag_test(text_input):
    
    tag_list = load_voca(DATA_NAME='total',METHOD='tag',DATATYPE='train',DETAIL='voca')
    status_logging("tag preprocessing(test) start ")
    mlb = MultiLabelBinarizer(classes=tag_list)
    tag_val=mlb.fit_transform(text_input['tag'])
    status_logging("tag preprocessing(test) end ")

    save_feature(DATA_NAME='total',METHOD='tag',DATATYPE='test',DETAIL='test',res=tag_val)

    return