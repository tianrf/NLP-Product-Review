from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding,Conv1D,LSTM,Dropout,Dense,Activation,GlobalMaxPool1D
import joblib


embeddings=joblib.load('embedding.pkl')

def pretrained_embedding(embeddings):
    vocab_num=embeddings.shape[0]

    embedding_dim=embeddings.shape[1]

    embedding_layer=Embedding(vocab_num,embedding_dim,trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([embeddings])

    return embedding_layer

def sentiment_model(input_dim, embeddings):

    sentence_indices=Input(shape=input_dim,dtype='int32')

    embedding_layer=pretrained_embedding(embeddings)

    word_vec=embedding_layer(sentence_indices)

    X=Conv1D(filters=200,
               kernel_size=[3],
               strides=1,
               padding='same',
               activation='relu')(word_vec)

    X=LSTM(128,return_sequences=True)(X)

    X=Dropout(0.5)(X)

    X=LSTM(128)(X)

    X=Dropout(0.5)(X)

    X=Dense(3)(X)

    X=Activation('softmax')(X)

    model=Model(inputs=sentence_indices,outputs=X)

    return model




