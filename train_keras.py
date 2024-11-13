import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

titanic_df = pd.read_csv("titanic.csv").dropna(subset=['Age'])
x = titanic_df[["Age", "Pclass"]].to_numpy(dtype=float)
y = titanic_df["Survived"].to_numpy()

model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(2, use_bias=True, activation="relu"))
model.add(Dense(2, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=True, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="sgd")
model.summary()

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())

model.fit(
    x=x,
    y=y,
    batch_size=9,
    epochs=35,
    verbose=1,
    validation_split=0.3,
    shuffle=True
)

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())

