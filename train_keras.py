import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data_path = 'titanic.csv'  
titanic_data = pd.read_csv(data_path)
titanic_data = titanic_data[['Survived', 'Pclass', 'Age']].dropna()
X = titanic_data[['Pclass', 'Age']]
y = titanic_data['Survived']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(8, input_shape=(2,), activation='relu', kernel_initializer=GlorotUniform(seed=0)),
    Dense(4, activation='relu', kernel_initializer=GlorotUniform(seed=0)),
    Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=0))
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

initial_weights = [layer.get_weights() for layer in model.layers]
print("Initial Weights:", initial_weights)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

final_weights = [layer.get_weights() for layer in model.layers]
print("Final Weights:", final_weights)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss During Training')
plt.show()

convergence_threshold = 0.001  
last_epochs_to_check = 10  

loss_diff = [abs(history.history['loss'][i] - history.history['loss'][i-1]) 
             for i in range(1, len(history.history['loss']))]

converged = False
converged_epoch = None

if len(history.history['loss']) > last_epochs_to_check:
    recent_loss_diff = loss_diff[-last_epochs_to_check:]
    avg_recent_loss_diff = sum(recent_loss_diff) / len(recent_loss_diff)
    if avg_recent_loss_diff < convergence_threshold:
        converged = True
        converged_epoch = len(history.history['loss']) - last_epochs_to_check

if converged:
    print(f"The model converged after approximately {converged_epoch} epochs.")
else:
    print("The model did not converge within the specified epoch limit.")

