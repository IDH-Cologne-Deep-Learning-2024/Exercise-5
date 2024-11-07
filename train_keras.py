import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')

data = data[['Age', 'Pclass', 'Survived']]

data = data.dropna()

X = data[['Age', 'Pclass']].values
y = data['Survived'].values

X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),  
    Dense(4, activation='relu'),                    
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

initial_weights = [layer.get_weights() for layer in model.layers]

history = model.fit(X, y, epochs=100, batch_size=32, verbose=1)

final_weights = [layer.get_weights() for layer in model.layers]

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.show()

for i, (initial, final) in enumerate(zip(initial_weights, final_weights), start=1):
    print(f"\nLayer {i} - Initial Weights:\n", np.array(initial[0]))
    print(f"Layer {i} - Final Weights:\n", np.array(final[0]))

loss_values = history.history['loss']
converged_epoch = np.argmin(np.diff(loss_values) < 1e-4) if any(np.diff(loss_values) < 1e-4) else "Did not converge significantly"
print(f"\nModel converged at epoch: {converged_epoch + 1 if isinstance(converged_epoch, int) else converged_epoch}")
