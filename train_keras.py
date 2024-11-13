# Import necessary libraries
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('titanic_data.csv')  # Adjust the path if necessary

# Step 2: Prepare the data
# Select relevant features and target variable
X = data[['Age', 'Pclass']]
y = data['Survived']

# Handle missing values in "Age" by filling with the mean
X['Age'].fillna(X['Age'].mean(), inplace=True)

# Step 3: Define the neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),  # Input layer + first hidden layer
    Dense(8, activation='relu'),                     # Second hidden layer
    Dense(1, activation='sigmoid')                   # Output layer for binary classification
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Inspect initial weights
initial_weights = [layer.get_weights() for layer in model.layers]
print("Initial Weights:")
for i, weights in enumerate(initial_weights):
    print(f"Layer {i + 1} weights:", weights)

# Step 5: Train the model and record training history
history = model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# Step 6: Inspect weights after training
trained_weights = [layer.get_weights() for layer in model.layers]
print("\nTrained Weights:")
for i, weights in enumerate(trained_weights):
    print(f"Layer {i + 1} weights:", weights)

# Step 7: Plot the training loss to check convergence
plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Additional check: Print number of epochs until loss stabilizes
final_loss = history.history['loss'][-1]
print(f"\nFinal Loss: {final_loss:.4f}")
