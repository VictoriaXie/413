import pandas as pd 
import numpy as np
import tensorflow as tensorflow 
from gensim.models import Word2Vec
import rnn

def embedding(tokens, model):
    embedding = [model.wv[token] for token in tokens if token in model.wv]
    return embedding

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation, Dropout, SpatialDropout1D

model = Sequential()
# vocab_size: distinct tokens len(w2v_model.wv)
# output_dim: dimension of the world2vec vectors we used

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_vector_length, input_length=10))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # use this for binary classificattion

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()


# train the model

# m = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

if __name__ == "__main__":
    human_token = pd.read_csv("")
    ai_token = pd.read_csv("ai_token.csv", index_col=0)
    model_w2v = Word2Vec.load("w2vmodel.model")

    token = pd.concat([human_token, ai_token])