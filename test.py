import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
df = pd.read_csv('/titanic.csv')  # Adjust path as necessary
df = df[['Age', 'Pclass', 'Survived']]  # Select relevant columns
df.dropna(subset=['Age'], inplace=True)  # Remove rows with missing age

# Features and target
X = df[['Age', 'Pclass']].values
y = df['Survived'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build the feed-forward neural network model
model = Sequential([
    Dense(8, input_shape=(2,), activation='relu'),  # Hidden layer with 8 neurons
    Dense(4, activation='relu'),                    # Hidden layer with 4 neurons
    Dense(1, activation='sigmoid')                  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Inspect initial weights
print("Initial Weights:")
for layer in model.layers:
    weights, biases = layer.get_weights()
    print("Layer weights:", weights)
    print("Layer biases:", biases)

# Step 4: Train the model and observe the loss over epochs
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Print the loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Check if the model has converged by observing the loss plot
final_epoch = len(history.history['loss'])
print(f"Model converged after {final_epoch} epochs.")

# Step 5: Inspect weights after training
print("\nWeights After Training:")
for layer in model.layers:
    weights, biases = layer.get_weights()
    print("Layer weights:", weights)
    print("Layer biases:", biases)
