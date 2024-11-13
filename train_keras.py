import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

titanic = pd.read_csv('titanic.csv')
titanic = titanic.dropna()
x = titanic[["Age", "Pclass"]]
y = titanic["Survived"]

FFNN = Sequential()
FFNN.add(Input(shape=(2,)))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(1, use_bias=True, activation='sigmoid'))
FFNN.compile(loss='binary_crossentropy', optimizer="sgd")
print("initial weights: \n" + str(FFNN.layers[0].get_weights()))
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())
history = FFNN.fit(x=x, y=y, epochs=300, verbose=0, validation_split=0.2)
print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())

loss = history.history['loss']
val_loss = history.history['val_loss']
converged_epochs = np.argmin(val_loss) + 1
print("Final Training loss: \n" + str(loss) + "\nFinal Validation loss: \n" + str(val_loss)
      + "\nThe model converged after "+str(converged_epochs)+" epochs")
 