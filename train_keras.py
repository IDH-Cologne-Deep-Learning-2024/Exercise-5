# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('/Users/hasanatici/Exercise-5/titanic.csv')  
features = data[['Age', 'Pclass']]
target = data['Survived']

# missing values in Age
features['Age'].fillna(features['Age'].median(), inplace=True)

#test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# neural network model
model = Sequential([
    Dense(16, input_shape=(2,), activation='relu'),  # Input layer
    Dense(8, activation='relu'),                     # Hidden layer
    Dense(1, activation='sigmoid')                   # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# initial weights before training
print("Initial weights before training:")
for layer in model.layers:
    print(f"Layer: {layer.name}, Initial Weights: {layer.get_weights()}")

# train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

# updated weights after training
print("\nUpdated weights after training:")
for layer in model.layers:
    print(f"Layer: {layer.name}, Updated Weights: {layer.get_weights()}")

#  Plot the training and validation loss to analyze convergence
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
