import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

df = pd.read_csv("titanic.csv", header=0)

model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer="sgd")

# model.summary()

x = df[["Age", "Pclass"]].values
y = df["Survived"].values

for layer in model.layers:
    w, _ = layer.get_weights()
    print(f"{w}")

model.fit(x=x,y=y,epochs=4,batch_size=200)


for layer in model.layers:
    w, _ = layer.get_weights()
    print(f"{w}")