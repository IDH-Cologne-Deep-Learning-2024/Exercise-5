import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

df = pd.read_csv('titanic.csv')
clean_df = df.dropna()
x = clean_df[["Age", "Pclass"]]
y = clean_df["Survived"]

FFNN = Sequential()
FFNN.add(Input(shape=(2,)))
FFNN.add(Dense(3, use_bias=True, activation='relu'))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(1, use_bias=True, activation='sigmoid'))
FFNN.compile(loss='binary_crossentropy', optimizer="sgd")

FFNN.fit(x = x, y = y, epochs = 100, verbose = 2, validation_freq=3)

print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())

FFNN.summary()
