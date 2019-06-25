import _pickle as cPickle
import matplotlib.pyplot as plt

file_list=['total']

class DeniedTagError(Exception):
    def __str__(self):
        return "Tag should be in here: '정치','경제','사회','세계','과학'"

def len_measure(series,type_name):
    if type_name =='MAX':
        return max(series.map(lambda x: len(str(x))))
    elif type_name == 'MIN':
        return min(series.map(lambda x: len(str(x))))
    elif type_name == 'AVG':
        return sum(series.map(lambda x: len(x)))/series.size


def tag_distn(series):
    tag_list=set(['정치','경제','사회','세계','과학'])
    tag_distin=dict()
    tag_merge={'정치':0,'경제':0,'사회':0,'세계':0,'과학':0}
    for i in series:
        if not i.issubset(tag_list):
            raise DeniedTagError
        #멀티태그를 하나로 태그로 간주했을 때
        try:
            tag_distin[str(i)]+=1
        except KeyError:
            tag_distin[str(i)]=1
        #각 태그별 축적
        for j in i:
            tag_merge[j]+=1
    
    return tag_distin,tag_merge
    

def data_stat():
    for file_name in file_list:
        with open('crawling/puri_news_data_{FILE_NAME}.pkl'.format(FILE_NAME=file_name),'rb') as infile:
            news_dset=cPickle.load(infile)
        
        with open('crawling/statistic_{FILE_NAME}.txt'.format(FILE_NAME=file_name),'w') as outfile:
            format_text="{BODY:<10}{MAX:^10}{MIN:^10}{AVG:^10}\n"
            outfile.write(format_text.format(BODY='',MAX='MAX',MIN='MIN',AVG='AVG'))
            for body_name in ['title','body']:
                outfile.write(format_text.format(BODY=body_name,
                                        MAX=len_measure(news_dset[body_name],'MAX'),
                                        MIN=len_measure(news_dset[body_name],'MIN'),
                                        AVG=len_measure(news_dset[body_name],'AVG')))
            outfile.write("\n{BODY:<10}\nkind of tag: '정치','경제','사회','세계','과학'\
                            \nCardinality: {VAL:>30}\n".format(BODY='tag',VAL=len_measure(news_dset['tag'],'AVG')))
            for i in tag_distn(news_dset['tag']):
                outfile.write(str(i))
                outfile.write('\n')

def data_distn_extraction():
    for file_name in file_list:
        with open('crawling/puri_news_data_{FILE_NAME}.pkl'.format(FILE_NAME=file_name),'rb') as infile:
            news_dset=cPickle.load(infile)['tag']
        
        tag_count=tag_distn(news_dset)

        for n,rst in enumerate(tag_count):
            group_data=list(rst.values())
            group_names=list(rst.keys())
            plt.bar(group_names,group_data)

            fig=plt.gcf()
            fig.savefig('data_distn_{0}.png'.format(n))

            



if __name__ == '__main__':
    data_stat()
    #data_distn_extraction()

        
        