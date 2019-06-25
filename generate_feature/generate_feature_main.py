
from generate_feature_tfidf_matrix import generate_tfidf_train,generate_tfidf_test
from generate_feature_fileio import read_files
from generate_tag import save_tag_train,save_tag_test
from log_ import error_logging, status_logging

def wrapper_generate_feature(train_input,test_input):

    generate_tfidf_train(train_input)
    generate_tfidf_test(test_input)

    save_tag_train(train_input)
    save_tag_test(test_input)
    return 

def main():
    from sklearn.model_selection import train_test_split

    total_data = read_files('total')
    
    train_text_input,test_text_input = train_test_split(total_data,test_size=0.3,random_state=0)# for generating feature 
    wrapper_generate_feature(train_text_input,test_text_input)

    return

if __name__=='__main__':
    try:
        main()
    except:
        error_logging()
        pass