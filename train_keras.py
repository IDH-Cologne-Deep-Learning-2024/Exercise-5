import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

FFNN = Sequential()
FFNN.add(Input(shape=(4,)))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(2, use_bias=True, activation='relu'))
FFNN.add(Dense(1, use_bias=True, activation='sigmoid'))

FFNN.compile(loss='binary_crossentropy', optimizer="sgd")
FFNN.summary()

print(FFNN.layers[0].get_weights())
print(FFNN.layers[1].get_weights())
print(FFNN.layers[2].get_weights())
[array([[ 0.350904 , 0.6731 ],
        [-0.701056 , -0.9477575 ],
        [ 0.38255095 , 0.81136656],
        [-0.8567922 , -0.42808342]], dtype=float32), array([0., 0.], dtype=float32)]
[array([[-0.67425936 , -0.43325353],
        [-0.5687941 , -0.61114633]], dtype=float32), array([0., 0.], dtype=float32)]
[array([[ 1.2658712 ],
        [-0.93679833]], dtype=float32), array([0.], dtype=float32)]