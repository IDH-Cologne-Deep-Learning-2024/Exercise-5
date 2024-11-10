import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1. Load data
data = pd.read_csv('titanic.csv')

# 2. Select relevant features (Pclass and Age) and the label (Survived)
# Fill missing values in 'Age' with the mean
data['Age'].fillna(data['Age'].mean(), inplace=True)
X = data[['Pclass', 'Age']]
y = data['Survived']

# 3. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the model
model = Sequential([
    Dense(10, input_shape=(2,), activation='relu'),  # Hidden layer with 10 neurons
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# 6. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# 8. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')

# 9. Examine the weights before and after training
print("\nWeights after training:")
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    print(f'Layer {i}: Weights after training')
    print(weights)
    print(f'Layer {i}: Bias after training')
    print(biases)