from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Activation, Dropout


def pretrained_embedding(embeddings):
    vocab_length = embeddings.shape[0]
    vec_dimension = embeddings.shape[1]

    embedding_layer = Embedding(input_dim=vocab_length, output_dim=vec_dimension,trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([embeddings])
    return embedding_layer

def sentimental_model(Input_shape, embeddings):

    sentence_indices = Input(shape=Input_shape,dtype='int32')

    X = pretrained_embedding(embeddings)(sentence_indices)

    X = Bidirectional(LSTM(128, return_sequences=True))(X)

    X = Dropout(rate=0.5)(X)

    X = Bidirectional(LSTM(128))(X)

    X = Dropout(rate=0.5)(X)

    X = Dense(units=3)(X)

    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model

