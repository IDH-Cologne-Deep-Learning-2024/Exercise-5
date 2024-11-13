import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Input
from keras.activations import *
import pandas as pd

dataladen = pd.read_csv("titanic.csv")
data = dataladen.dropna()
x_data = data[["Age", "Pclass"]]
y_data = data[["Survived"]]


FFNN = Sequential ()
FFNN.add(Input(shape=(2,)))
FFNN.add(Dense(3, use_bias=True , activation='relu'))
FFNN.add(Dense(2, use_bias=True , activation='relu'))
FFNN.add(Dense(1, use_bias=True , activation='sigmoid'))
FFNN.compile(loss='binary_crossentropy', optimizer="sgd")

FFNN.fit(x = x_data, y = y_data, epochs = 250)

print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())

FFNN.summary ()