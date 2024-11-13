import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
df = df[["Age", "Pclass", "Survived"]].dropna()

x = df[["Age", "Pclass"]].values
y = df["Survived"].values

# initialize FFNN model
FFNN = Sequential(
    [
        Input(shape=(2,)),
        Dense(10, use_bias=True, activation="relu"),
        Dense(6, use_bias=True, activation="relu"),
        Dense(1, use_bias=True, activation="sigmoid"),
    ]
)

FFNN.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# saving weights before training
weights_before = []
for i, layer in enumerate(FFNN.layers):
    weights = layer.get_weights()
    weights_before.append(weights)

# training the model
FFNN_history = FFNN.fit(
    x, y, epochs=250, batch_size=64, validation_split=0.2, verbose=0
)

# saving weights after training
weights_after = []
for i, layer in enumerate(FFNN.layers):
    weights = layer.get_weights()
    weights_after.append(weights)

# printing weights
print("Weights: \n")
for i, (before, after) in enumerate(zip(weights_before, weights_after)):
    print(f"Layer {i}: \n")
    print(f"Before:\n{before[0]}")
    print(f"After:\n{after[0]}")
    print()

# plot
plt.plot(FFNN_history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss while training")
plt.show()

# check for conversion
val_loss = FFNN_history.history["loss"]
converged_epoch = next(
    (i for i in range(1, len(val_loss)) if abs(val_loss[i] - val_loss[i - 1]) < 1e-4),
    None,
)
print(
    f"The model converged at epoch {converged_epoch + 1}."
    if converged_epoch is not None
    else "The model did not converge."
)
