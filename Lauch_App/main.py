from tensorflow.keras.models import load_model
import joblib
import jieba
import numpy as np
from flask import Flask, render_template, session, url_for,redirect
from wtforms import StringField,SubmitField
from flask_wtf import FlaskForm
from model3 import *

max_len=100


################################################################
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
#######################################################################################


def return_model_prediction(model,sentnece_processor,text):
    review_list=['差评','中评','好评']
    split_word=sentence_processor.split_sentence(text)
    word_indices=sentence_processor.sentence_to_index(split_word)
    review_index=model.predict(word_indices).argmax()

    return (review_list[review_index])
########################################################################################



app=Flask(__name__)
app.config['SECRET_KEY']= 'Emiya_Shiro'



review_model = sentiment_model(max_len,embeddings)
review_model.load_weights('Conv1d-LSTM.h5')
index2word = joblib.load('index2word.pkl')
word2index = joblib.load('word2index.pkl')
sentence_processor = sentence_process(index2word,word2index,max_len)

class RviewForm(FlaskForm):
    text = StringField('评价内容')
    submit = SubmitField('提交')



@app.route('/', methods=['GET','POST'])
def index():
    form = RviewForm()

    if form.validate_on_submit():
        session['text']=form.text.data

        return redirect(url_for('prediction'))
    return render_template('home.html',form=form)


@app.route('/prediction')
def prediction():
    text=session['text']
    results=return_model_prediction(review_model,sentence_processor,text)

    return render_template('prediction.html',results=results)


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)














