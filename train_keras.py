import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

titanic = pd.read_csv("titanic.csv")

titanic = titanic[['Age', 'Pclass', 'Survived']].dropna()
X = titanic[['Age', 'Pclass']].values #featrures
Y = titanic['Survived'].values #target

model = Sequential([
    Input(shape =(2,)),
    Dense(3, activation='relu'),
    Dense(2, activation="relu"),
    Dense(1, activation="sigmoid")])

model.compile(loss='binary_crossentropy', optimizer="sgd")

print("Initial weights of each layer:")
for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"Weights: {weights}, Biases: {biases}")

history = model.fit(X, Y, epochs=200, batch_size=64, validation_split=0.2, verbose=1)

print("\nWeights after training:")
for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"Weights: {weights}, Biases: {biases}")
  

loss = history.history['loss']
val_loss = history.history['val_loss']
print("\nFinal Training Loss:", loss[-1])
print("Final Validation Loss:", val_loss[-1])

converged_epochs = np.argmin(val_loss) + 1  # Index of minimum val_loss, +1 to count epochs
print(f"The model converged after {converged_epochs} epochs with the minimum validation loss of {min(val_loss)}.")