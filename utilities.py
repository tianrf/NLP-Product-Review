
import numpy as np
import jieba
import csv
import gensim
from gensim.models.word2vec import Word2Vec

#read csv
##########################################################
def read_csv(pathfile,header=True):
    phrase = []
    label = []

    with open (pathfile) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            label.append(row[1])
    if header:
        X = np.asarray(phrase[1:])
        Y = np.asarray(label[1:], dtype=int)
    else:
        X = np.asarray(phrase)
        Y = np.asarray(label, dtype=int)
    return X,Y
###########################################################

def load_word2vec():
    w2v_model=gensim.models.KeyedVectors.load("Tencent_Word_100W.bin",mmap='r')
    w2v_model.index2entity[0]=' '

    word2vec=w2v_model

    word_embedding=w2v_model.vectors.copy()
    word_embedding[0]=np.zeros(200)

    index2word=w2v_model.index2word

    word2index={token:index for index,token in enumerate(index2word)}

    return word_embedding,word2vec,index2word,word2index


############################################################
def one_hot_encoding(Y):
   K=len(set(Y))
   Y_one_hot=np.eye(K)[Y]
   return Y_one_hot

###########################################################
class sentence_process():
    def __init__(self,index2word,word2index,max_len=200):
        self.index2word=index2word
        self.word2index=word2index
        self.max_len=max_len
        jieba.load_userdict(index2word)


    def split_sentence(self,text):
        if isinstance(text,str):
            split_list=[word for word in jieba.lcut(text,cut_all=False) if word!=' ']
            return np.array(split_list).reshape(1, -1)
        else:
            split_list=[jieba.lcut(sentence,cut_all=False) for sentence in text]
            for i in range(len(split_list)):
                split_list[i]=[word for word in split_list[i] if word!=' ']
                i+=1
            return np.array(split_list)


    def sentence_to_index(self,split_list):
        m=split_list.shape[0]
        X=np.zeros((m,self.max_len),dtype=int)

        for i in range(m):
            j=0
            for word in split_list[i,]:
                try:
                    X[i][j]=self.word2index[word]
                except:
                    X[i][j]=self.word2index['unknown']

                j=j+1
                if j==self.max_len:
                    break
        return(X)



