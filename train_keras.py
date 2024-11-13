import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
# plot output
import numpy as np
import matplotlib.pyplot as plt


titanic_data = pd.read_csv('titanic.csv')

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)  # I filled missing ages with the median
features = titanic_data[['Pclass', 'Age']]
target = titanic_data['Survived']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

#NN
model = Sequential([
    Input(shape=(2,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


initial_weights = model.get_weights()


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, batch_size=32, verbose=1)


final_weights = model.get_weights()

print("\nInitial weights:\n", initial_weights)
print("\nFinal weights:\n", final_weights)

#Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

#Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTraining Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# plot output --> Ignore

ages = np.linspace(0, 80, 100)

predictions = {}

# predictions
for pclass in [1, 2, 3]:
    input_data = np.array([[pclass, age] for age in ages])
    input_data_scaled = scaler.transform(input_data)  

    survival_probabilities = model.predict(input_data_scaled).flatten()

    predictions[f'Class {pclass}'] = survival_probabilities

plt.figure(figsize=(10, 6))
for pclass, survival_probabilities in predictions.items():
    plt.plot(ages, survival_probabilities, label=f'{pclass}')

plt.xlabel('Age')
plt.ylabel('Predicted Survival Probability')
plt.title('Predicted Survival Probability by Age and Passenger Class')
plt.legend(title='Passenger Class')
plt.grid()
plt.show()