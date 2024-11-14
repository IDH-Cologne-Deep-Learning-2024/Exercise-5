import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


df = pd.read_csv("titanic.csv")
df = df.dropna()

# Init the model
FFNN = Sequential()
FFNN.add(Input(shape=(2,)))  # Shape needs to correspond to number of features
FFNN.add(Dense(2, activation="relu"))
FFNN.add(Dense(2, activation="relu"))
FFNN.add(Dense(1, activation="sigmoid"))
FFNN.compile(loss='binary_crossentropy', optimizer="sgd")
print("Random weights and biases assigned when model gets initialized")
print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())

# Fit the model to the data
FFNN.fit(df[["Age", "Pclass"]], df["Survived"],
         epochs=20,
         initial_epoch=0,
         verbose=0)
# The weights are now trained accordingly
print("Weights and biases after training")
print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())
