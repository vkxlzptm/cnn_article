# 주어진 데이터를 최대한 고르게 섞을려고 노력했음
'''
각 테그가 300개 정도는 있도록 노력하고 cardinality도 확보하기 위해 노력함
joongang은 주로 단일테그 추출용으로 newsis는 멀티테그 추출용으로 확

joongang
- (사회) 310개 제거
- (경제) 90개 제거
- (정치) 146개 제거
- (세계) 40개 제거
joongang_IT
- (과학) 80개 제거
newsis
- (경제) 전부 제
- (사회) 전부 제거
- (정치) 전부 제거
newsis_world
- (세계) 전부 제거
newsis_IT
- (과학, 경제): 100개 제거

데이터 크기 제한
- title: 15자 이상
- body: 300자 이상
'''
import pandas as pd
import _pickle as cPickle

def len_remover(df_data):
    
    x=df_data[df_data['title'].map(lambda x: len(x))>=16]
    return x[x['body'].map(lambda x: len(x))>=300].reset_index(drop=True)

def tag_remover(df_data,tag,cdt):
    print(tag,cdt)###
    if cdt =='all':
        return df_data[df_data['tag']!=tag ].reset_index(drop=True)
    else:
        if not isinstance(cdt,int):
            raise Exception
        drop_loc=list()
        # 주의: 주어진 자료보다 더 많은 테그를 삭제하려 하면 오류남
        for i in range(df_data.shape[0]):
            try:
                if df_data.at[i,'tag'] == tag:
                    drop_loc.append(i)
            except:
                continue

            if len(drop_loc)>=cdt:
                return df_data.drop(drop_loc).reset_index(drop=True)
            else:
                continue
    print('Number of deleted data is {0}, but you  require {1}.It the maximal deleting size'.format(len(drop_loc),cdt))
    return df_data.drop(drop_loc).reset_index(drop=True)

def main():
    file_name_format='crawling/puri_news_data_{FILE_NAME}.pkl'
    tag_remover_set={'joongang':((('사회',),310),(('경제',),90),(('정치',),146),(('세계',),40)),
                     'joongang_IT':((('과학',),100),),
                     'newsis':((('경제',),'all'),(('사회',),'all'),(('정치',),'all')),
                     'newsis_world':((('세계',),'all'),),
                     'newsis_IT':((('과학','경제'),100),),
                     'digitaltimes':((('경제',),6500),(('과학',),500))}
    with open(file_name_format.format(FILE_NAME='total'),'wb') as outfile:
        stored_file=pd.DataFrame(columns=['title','body','tag'])
        for file_name, remover_list in tag_remover_set.items():
            with open(file_name_format.format(FILE_NAME=file_name),'rb') as infile:
                news_dset=len_remover(cPickle.load(infile))
                print(file_name,news_dset.shape)####
                for tag,cond in remover_list:
                    news_dset=tag_remover(news_dset,set(tag),cond)
                stored_file=pd.concat([stored_file,news_dset])
        cPickle.dump(stored_file,outfile,-1)
    with open('puri_news_data_ex_total.txt','w') as outfile:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            outfile.write('data lengh: '+str(news_dset.shape[0])+'\n'+'_'*40+'\n')
            outfile.write(str(stored_file.head(20)))
                
                

if __name__ == "__main__":
    main()

