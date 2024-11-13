import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

def read_data(path):
    dataframe=pd.read_csv(path, header=0).dropna()
    return dataframe

FFNN = Sequential ()
FFNN.add(Input(shape=(4,)))
FFNN.add(Dense(2, use_bias=True , activation='relu'))
FFNN.add(Dense(4, use_bias=True , activation='softmax'))
FFNN.add(Dense(2, use_bias=True , activation='relu'))
FFNN.add(Dense(1, use_bias=True , activation='sigmoid '))

FFNN.compile(loss='binary_crossentropy ', optimizer="sgd")

FFNN.summary ()

data=read_data("titanic.csv")

FFNN.fit(x=data["Age"], y=data["Survived"], )
