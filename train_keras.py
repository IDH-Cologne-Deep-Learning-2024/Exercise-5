import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt


df = pd.read_csv('titanic.csv') 
df = df[['Age', 'Pclass', 'Survived']].dropna() 

# Separate features and target
X = df[['Age', 'Pclass']].values  
y = df['Survived'].values

# Manual train/test split (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


FFNN = Sequential()
FFNN.add(Input(shape=(2,)))  
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(1, use_bias=True, activation='sigmoid'))

# Compile the model
FFNN.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

# Display model summary and initial weights
FFNN.summary()
print("\nInitial Weights:")
for i, layer in enumerate(FFNN.layers):
    weights, biases = layer.get_weights()
    print(f"Layer {i} weights:", weights)
    print(f"Layer {i} biases:", biases)

# Train the model and capture training history
history = FFNN.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Plot the loss over epochs to observe convergence
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# Check if the model has converged by observing the loss plot
final_epoch = len(history.history['loss'])
print(f"\nModel converged after {final_epoch} epochs.")

# Display weights after training
print("\nWeights After Training:")
for i, layer in enumerate(FFNN.layers):
    weights, biases = layer.get_weights()
    print(f"Layer {i} weights:", weights)
    print(f"Layer {i} biases:", biases)
