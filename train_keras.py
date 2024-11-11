import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
    dataframe = pd.read_csv(path, sep=",")
    return dataframe

#building model
def build():
    model = Sequential([
        Input(shape= (2,)),
        Dense(4, activation= 'relu'), 
        Dense(2, activation= 'relu'),
        Dense(1, activation= 'sigmoid')
    ])
    model.compile(optimizer='sgd', loss= 'binary_crossentropy')
    return model

#plot changing loss
def plotten(history):
    plt.plot(history.history['loss'], label= 'Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def main():
    #setting up df
    dataframe = read_data('titanic.csv')
    dataframe = dataframe[dataframe['Age'].notna()]
    dataframe['Age'] = dataframe['Age'] / 10 #shouldve standardized instead? or in addition
    x = dataframe[['Age', 'Pclass']].values
    y = dataframe['Survived'].values

    model = build()

    #pre-training
    print("Weights pre-T: ")
    for layer in model.layers:
        weights= layer.get_weights()
        print(weights)

    #fit to train     
    history= model.fit(x, y, epochs= 113, batch_size=23, verbose=1)

    #post-training
    print("Weights post-T: ")
    for layer in model.layers:
        weights= layer.get_weights()
        print(weights)

    plotten(history)

main()

    