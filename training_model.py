from utilities import *
from model import *
from tensorflow.keras.optimizers import Adam


max_len=100

X_train,Y_train=read_csv('data/train_data.csv')
X_test,Y_test=read_csv('data/test_data.csv')

Y_train=one_hot_encoding(Y_train)
Y_test=one_hot_encoding(Y_test)


word_embeddings,word2vec,index2word,word2index=load_word2vec()

sen_processor=sentence_process(index2word,word2index,max_len=max_len)

X_train_split=sen_processor.split_sentence(X_train)
X_train_indices=sen_processor.sentence_to_index(X_train_split)

X_test_split=sen_processor.split_sentence(X_test)
X_test_indices=sen_processor.sentence_to_index(X_test_split)



model=sentimental_model(max_len,word_embeddings)

opt=Adam(learning_rate=0.005)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics='accuracy')

model.fit(x=X_train_indices,y=Y_train,batch_size=64,epochs=50)

model.evaluate(x=X_test_indices,y=Y_test)