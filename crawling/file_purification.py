'''
#제목/본문 이슈
- [],(),【】에 쓸데없는 정보가 들어간 경우가 많음
- 한자 사용
-___ 기자 라는 말이 존재함
- 이메일이 들어가 있기도 함

- 상단에 예시가 된 괄호는 제거 대상
- hanja를 사용해서 한자 번역
- '기자'라는 말을 발견 시 이전 한 글자 삭제(이름이라고 추정)
- 그 이외에 한글이 아닌 부분은 전부 제거
# 태그 이슈
-태그는 정치,경제,사회,세계,IT로 제한
-IT 태그 중 과학과 관련된 것도 있음
-집합으로 태그를 모으는 것이 활용에 편함

-해당 테그에 속하지 않는 데이터 파일은 전부 삭제(멀티 태그 포함)
-IT는 포괄성을 위해 과학 태그로 변환
-집합 태그로 변환
# 기타 이슈
-필요없는 url데이터 제거
'''

file_list=['digitaltimes']

import pandas as pd
import _pickle as cPickle
import re
import hanja

# vs code 환경으로 인해 일부러 넣은 코드()
import os
if 'crawling' not in os.getcwd():
    os.chdir(os.getcwd()+'/crawling')
# -------------------

def main():
    #url 태그 없이 data 불러오기
    for file_name in file_list:
        news_file_name='news_data_{FILE_NAME}.pkl'.format(FILE_NAME=file_name)
        with open(news_file_name,'rb') as infile:
            news_dset=cPickle.load(infile).loc[:,['title','body','tag']]
        #본문/제목 이슈
        first_fun=lambda x: re.sub(r'\[.*?\]|\(.*?\)|【.*?】','',str(x))  #괄호 제거(괄호 안의 괄호는 없다고 가정)
        second_fun=lambda x: hanja.translate(str(x),'substitution')         #한자 제거
        third_fun=lambda x: re.sub(r'[가-힣]*\s+(기자|특파원)','',str(x))  #기자 이름 제거
        fourth_fun=lambda x: re.sub(r'[^가-힣]|\s+','',str(x))                  #그 이외 한글이 아닌 부분 제거
        news_dset.loc[:,['title','body']]=news_dset.loc[:,['title','body']].applymap(first_fun).applymap(second_fun).applymap(third_fun).applymap(fourth_fun)

        #태그 이슈
        #쓸데 없는 태그가 들어간 부분을 제거
        removing_list=[]
        for i in range(news_dset.shape[0]):
            for j in news_dset.at[i,'tag']:
                if j in ['정치','경제','사회','세계','IT']:
                    continue
                else:
                    removing_list.append(i)
                    break
        news_dset=news_dset.drop(removing_list)
        
        first_fun=lambda x: list(map(lambda k: '과학' if str(k)=='IT' else k,list(x)))  #IT 태그를 과학 태그로 변환
        second_fun=lambda x: set(x)                                                     #집합 태그
        news_dset['tag']=news_dset['tag'].map(first_fun).map(second_fun)
        #처리된 파일 저장
        puri_news_file_name='puri_news_data_{FILE_NAME}.pkl'.format(FILE_NAME=file_name)
        puri_news_file_stat_name='puri_news_data_ex_{FILE_NAME}.txt'.format(FILE_NAME=file_name)
        with open(puri_news_file_name,'wb') as outfile:
            cPickle.dump(news_dset, outfile,-1)
        with open(puri_news_file_stat_name,'w') as outfile:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                outfile.write('data lengh: '+str(news_dset.shape[0])+'\n'+'_'*40+'\n')
                outfile.write(str(news_dset.head(20)))

if __name__ == '__main__':
    main()



    