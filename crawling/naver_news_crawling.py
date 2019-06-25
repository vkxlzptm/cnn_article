from bs4 import BeautifulSoup
import requests
import re

# 출력 파일 명

news_name='digitaltimes'
OUTPUT_FILE_NAME = 'news_data_{}.pkl'.format(news_name)
EXAMPLE_FILE_NAME='news_data_example_{}.txt'.format(news_name)
# 긁어 올 URL
URL_FOM = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=105&oid=029&aid=000'
URL_NUM=2491841

# 크롤링 함수
def tag_remover(text):
    #javascrip 태그 지우기
    js_reg=re.compile('<script.*script>',re.DOTALL)
    js_removed_text=re.sub(js_reg,'',text)
    #기자 정보 제거: 이메일의 @를 기준으로 제거한다. html 태그 정보도 활용하기 때문에 여기서 implementation을 함
    email_reg=re.compile('<br>[^<>]*@')
    try:
        trash_inf=email_reg.search(js_removed_text)
        html_text=js_removed_text[:trash_inf.start()]
    except:
        html_text=js_removed_text
    #html 태그를 모두 제거한다. 단 제거 중에 본문이 사라지지 않도록 해야한다.
    html_reg=re.compile(r'<[^<]*>|\<|\>')
    usual_text=re.sub(html_reg,'',html_text)
    #특수 테그를 제거한다.
    meta_reg=re.compile(r'\r|\n|\t')
    pure_text=re.sub(meta_reg,'',usual_text)
    # 리턴 전에 양쪽 공백을 제거한다
    return pure_text.strip()
    
    

def news_extraction(URL):
    # 한 뉴스에서 본문과 제목을 추출한다
    # <!-- 본문 내용 --> 이라는 부분이 있다는 가정하에서 코드를 짠다
    raw_file=requests.get(url=URL).text
    try:
        rough_data=[raw_file.split('title>')[-2],
                    re.split('본문 내용 -->',raw_file)[-2],
                    re.split('이 기사는 언론사에서|섹션으로 분류했습니다.',raw_file)[-2]]
    except:
        print(URL)
        raise Exception
    rough_data=list(map(tag_remover,rough_data))
    purified_data=list()
    # 제목 추출 ex> "정부에 하소연만 39번"…박용만 '규제 성토' 20분 : 네이버 뉴스
    purified_data.append(rough_data[0].split(':')[0].strip())
    # 본문 추출
    purified_data.append(rough_data[1])
    # 태그 추출
    purified_data.append(re.split(',',rough_data[2].replace(' ','')))
    return purified_data
    

# 메인 함수
import pandas as pd
import _pickle as cPickle
def main():
    #dataframe 형성
    po_n,ec_n,so_n,wo_n,IT_n=(0,0,0,0,0)
    data_list=list()
    i=-1
    while IT_n<10000 and wo_n<10000 and po_n<10000 and ec_n<10000 and so_n<10000:
        i+=1
        if i%5000 ==0:
            print('The {0}th state: {1}'.format(i,len(data_list)))
        URL=URL_FOM+str(URL_NUM-i)
        try:
            result_text=news_extraction(URL)
            # 각 태그를 만개 이상 추
            if 'IT' in result_text[2] and IT_n<10000:
                IT_n+=1
                pass
            elif '세계' in result_text[2] and wo_n<10000:
                wo_n+=1
                pass
            elif '정치' in result_text[2] and po_n<10000:
                po_n+=1
                pass
            elif '경제' in result_text[2] and ec_n<10000:
                ec_n+=1
                pass
            elif '사회' in result_text[2] and so_n<10000:
                so_n+=1
                pass
            else:
                continue
        except:# 삭제된 기사
            continue
        result_text.append(URL)
        data_list.append(result_text)
    news_data_set=pd.DataFrame(data_list,columns=['title','body','tag','url'])
    # dataframe 원본 저장
    with open(OUTPUT_FILE_NAME,'wb') as outfile:
        cPickle.dump(news_data_set,outfile,-1)
    # dataframe 예제 저장
    with open(EXAMPLE_FILE_NAME,'w') as outfile_ex:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            outfile_ex.write('data lengh: '+str(len(data_list))+'\n'+'_'*40+'\n')
            outfile_ex.write(str(news_data_set.head(20)))
 
if __name__ == '__main__':
    main()