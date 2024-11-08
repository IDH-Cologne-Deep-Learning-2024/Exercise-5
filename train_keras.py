import pandas as pd
import numpy as np
import keras as keras
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

dataframe = pd.read_csv("titanic.csv", sep=",")
dataframe.dropna()
FFNN = keras.Sequential()
FFNN.add(Input(shape=(2,)))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(1, use_bias=True, activation='sigmoid'))
FFNN.compile(loss='binary_crossentropy', optimizer="sgd")

print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())

modelhistory = FFNN.fit(x = dataframe[["Pclass", "Age"]], y = dataframe["Survived"], epochs=300)

print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())

plt.plot(modelhistory.history['loss'])
plt.show()

print("Minimum", min(modelhistory.history['loss']))
print("Minimum Epoch", np.argmin(modelhistory.history['loss']))
FFNN.summary()


