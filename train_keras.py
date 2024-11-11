import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

# use fit method in keras to train 
# s. Sequential model: single in-output stacks of layers; 
# look at weights of each layer before and after training and how they change
# plus look at loss and if model converges (if so, after how many epochs?)

#keras methods: compile, fit, evaluate, predict, train/test/predict_on_batch
#keras fit returns History obj 
#keras fit args: y, y, batch_size, epochs, verbosse, callbakcs, validation_split/data, shuffle, class/sample_weight, initial epoch, steps_per_epoch, validation_steps, validation_batch_size, _validation_freq

def read_data(path):
    dataframe = pd.read_csv(path, sep=",")
    return dataframe

def build(): #batch_input_shape?
    model = Sequential([
        Input(shape= (2,)),
        Dense(4, activation= 'relu'), 
        Dense(2, activation= 'relu'),
        Dense(1, activation= 'sigmoid')
    ])
    model.compile(optimizer='sgd', loss= 'binary_crossentropy')
    return model

def plotten(history):
    plt.plot(history.history['loss'], label= 'Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def main():
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
        
    history= model.fit(x, y, epochs= 113, batch_size=23, verbose=1)
    print("Weights post-T: ")
    for layer in model.layers:
        weights= layer.get_weights()
        print(weights)

    plotten(history)

main()

    