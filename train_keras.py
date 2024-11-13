import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')
X = data[['Age', 'Pclass']]
y = data['Survived']

# missing_values.handler
X['Age'].fillna(X['Age'].mean(), inplace=True)

# array.converter
X = X.values
y = y.values

# feature.normalizing
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# 70%train // 30%test
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# model.build
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# printing
print("Initial weights:")
for layer in model.layers:
    print(layer.get_weights())

# model.training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# printing.final
print("\nFinal weights:")
for layer in model.layers:
    print(layer.get_weights())

# plot.training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# model.evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Initial weights:
# [array([[ 0.01204211, -0.6966736 , -0.6059862 , -0.71330124, -0.51996195,
#          0.5108106 , -0.2530297 , -0.6189933 ],
#        [ 0.63503706,  0.5334625 ,  0.5369483 ,  0.36653566,  0.6428468 ,
#         -0.24268699, -0.4854105 , -0.70069766]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)]
# [array([[ 0.668608  ,  0.522656  , -0.34335384, -0.6548971 ],
#        [-0.38112062, -0.31737876, -0.60991293,  0.22051924],
#        [-0.12513739,  0.20011795, -0.65300184, -0.188725  ],
#        [ 0.4683985 , -0.66933495, -0.59643996,  0.39488178],
#        [-0.51115495,  0.14799833, -0.50701714, -0.3091687 ],
#        [-0.16804844,  0.15563077, -0.21286383, -0.5080358 ],
#        [-0.23763454,  0.1587773 , -0.49692988,  0.02852786],
#        [ 0.45401138,  0.52566   , -0.689762  ,  0.62924176]],
#       dtype=float32), array([0., 0., 0., 0.], dtype=float32)]
# [array([[ 0.5540811 ],
#        [ 0.78983784],
#        [-0.7936045 ],
#        [-0.383331  ]], dtype=float32), array([0.], dtype=float32)]
# Epoch 1/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 20ms/step - accuracy: 0.4530 - loss: 0.7077 - val_accuracy: 0.6160 - val_loss: 0.6988
# Epoch 2/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5625 - loss: 0.6968 - val_accuracy: 0.6240 - val_loss: 0.6927
# Epoch 3/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5291 - loss: 0.6983 - val_accuracy: 0.5920 - val_loss: 0.6889
# Epoch 4/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5528 - loss: 0.6956 - val_accuracy: 0.5920 - val_loss: 0.6852
# Epoch 5/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5827 - loss: 0.6848 - val_accuracy: 0.6320 - val_loss: 0.6814
# Epoch 6/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5820 - loss: 0.6831 - val_accuracy: 0.5920 - val_loss: 0.6783
# Epoch 7/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6626 - loss: 0.6712 - val_accuracy: 0.5920 - val_loss: 0.6752
# Epoch 8/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6377 - loss: 0.6748 - val_accuracy: 0.6320 - val_loss: 0.6731
# Epoch 9/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6775 - loss: 0.6756 - val_accuracy: 0.6480 - val_loss: 0.6708
# Epoch 10/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6980 - loss: 0.6674 - val_accuracy: 0.6640 - val_loss: 0.6688
# Epoch 11/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6716 - loss: 0.6628 - val_accuracy: 0.6640 - val_loss: 0.6666
# Epoch 12/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6562 - loss: 0.6680 - val_accuracy: 0.6720 - val_loss: 0.6647
# Epoch 13/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6485 - loss: 0.6697 - val_accuracy: 0.7040 - val_loss: 0.6630
# Epoch 14/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6932 - loss: 0.6609 - val_accuracy: 0.7040 - val_loss: 0.6607
# Epoch 15/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6728 - loss: 0.6650 - val_accuracy: 0.6960 - val_loss: 0.6592
# Epoch 16/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6794 - loss: 0.6621 - val_accuracy: 0.6960 - val_loss: 0.6571
# Epoch 17/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7193 - loss: 0.6556 - val_accuracy: 0.6960 - val_loss: 0.6559
# Epoch 18/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6822 - loss: 0.6628 - val_accuracy: 0.6960 - val_loss: 0.6547
# Epoch 19/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6844 - loss: 0.6602 - val_accuracy: 0.6960 - val_loss: 0.6531
# Epoch 20/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6838 - loss: 0.6578 - val_accuracy: 0.7040 - val_loss: 0.6514
# Epoch 21/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7323 - loss: 0.6413 - val_accuracy: 0.7040 - val_loss: 0.6501
# Epoch 22/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7059 - loss: 0.6497 - val_accuracy: 0.7040 - val_loss: 0.6487
# Epoch 23/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6770 - loss: 0.6521 - val_accuracy: 0.7040 - val_loss: 0.6477
# Epoch 24/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6735 - loss: 0.6542 - val_accuracy: 0.7040 - val_loss: 0.6462
# Epoch 25/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6736 - loss: 0.6495 - val_accuracy: 0.7040 - val_loss: 0.6450
# Epoch 26/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6642 - loss: 0.6522 - val_accuracy: 0.7040 - val_loss: 0.6443
# Epoch 27/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6701 - loss: 0.6502 - val_accuracy: 0.7040 - val_loss: 0.6432
# Epoch 28/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6873 - loss: 0.6410 - val_accuracy: 0.7040 - val_loss: 0.6418
# Epoch 29/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6784 - loss: 0.6410 - val_accuracy: 0.7040 - val_loss: 0.6407
# Epoch 30/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6770 - loss: 0.6439 - val_accuracy: 0.7040 - val_loss: 0.6401
# Epoch 31/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6906 - loss: 0.6413 - val_accuracy: 0.7040 - val_loss: 0.6393
# Epoch 32/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6693 - loss: 0.6412 - val_accuracy: 0.7040 - val_loss: 0.6378
# Epoch 33/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6692 - loss: 0.6441 - val_accuracy: 0.7040 - val_loss: 0.6373
# Epoch 34/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6576 - loss: 0.6389 - val_accuracy: 0.7040 - val_loss: 0.6363
# Epoch 35/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6740 - loss: 0.6460 - val_accuracy: 0.6960 - val_loss: 0.6356
# Epoch 36/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6797 - loss: 0.6410 - val_accuracy: 0.7040 - val_loss: 0.6345
# Epoch 37/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6780 - loss: 0.6410 - val_accuracy: 0.6960 - val_loss: 0.6337
# Epoch 38/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6678 - loss: 0.6383 - val_accuracy: 0.7040 - val_loss: 0.6324
# Epoch 39/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7050 - loss: 0.6237 - val_accuracy: 0.6960 - val_loss: 0.6322
# Epoch 40/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6751 - loss: 0.6439 - val_accuracy: 0.6960 - val_loss: 0.6318
# Epoch 41/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6913 - loss: 0.6317 - val_accuracy: 0.6960 - val_loss: 0.6304
# Epoch 42/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7025 - loss: 0.6319 - val_accuracy: 0.6960 - val_loss: 0.6306
# Epoch 43/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6754 - loss: 0.6374 - val_accuracy: 0.6960 - val_loss: 0.6296
# Epoch 44/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6896 - loss: 0.6323 - val_accuracy: 0.6960 - val_loss: 0.6290
# Epoch 45/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6746 - loss: 0.6403 - val_accuracy: 0.6960 - val_loss: 0.6281
# Epoch 46/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6589 - loss: 0.6425 - val_accuracy: 0.6960 - val_loss: 0.6277
# Epoch 47/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6629 - loss: 0.6383 - val_accuracy: 0.6960 - val_loss: 0.6271
# Epoch 48/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7059 - loss: 0.6237 - val_accuracy: 0.6960 - val_loss: 0.6264
# Epoch 49/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7069 - loss: 0.6152 - val_accuracy: 0.6960 - val_loss: 0.6251
# Epoch 50/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6663 - loss: 0.6312 - val_accuracy: 0.6960 - val_loss: 0.6249
# Epoch 51/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6763 - loss: 0.6338 - val_accuracy: 0.6960 - val_loss: 0.6247
# Epoch 52/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6693 - loss: 0.6316 - val_accuracy: 0.6960 - val_loss: 0.6240
# Epoch 53/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6823 - loss: 0.6342 - val_accuracy: 0.6960 - val_loss: 0.6237
# Epoch 54/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.6959 - loss: 0.6257 - val_accuracy: 0.6960 - val_loss: 0.6233
# Epoch 55/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6609 - loss: 0.6321 - val_accuracy: 0.6960 - val_loss: 0.6225
# Epoch 56/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7008 - loss: 0.6124 - val_accuracy: 0.6960 - val_loss: 0.6218
# Epoch 57/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6851 - loss: 0.6261 - val_accuracy: 0.6960 - val_loss: 0.6220
# Epoch 58/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6670 - loss: 0.6339 - val_accuracy: 0.6960 - val_loss: 0.6215
# Epoch 59/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6780 - loss: 0.6255 - val_accuracy: 0.6960 - val_loss: 0.6205
# Epoch 60/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6504 - loss: 0.6464 - val_accuracy: 0.6960 - val_loss: 0.6207
# Epoch 61/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6776 - loss: 0.6280 - val_accuracy: 0.6960 - val_loss: 0.6200
# Epoch 62/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7061 - loss: 0.6115 - val_accuracy: 0.6960 - val_loss: 0.6193
# Epoch 63/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6985 - loss: 0.6128 - val_accuracy: 0.6960 - val_loss: 0.6181
# Epoch 64/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6882 - loss: 0.6192 - val_accuracy: 0.6960 - val_loss: 0.6177
# Epoch 65/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6624 - loss: 0.6345 - val_accuracy: 0.6960 - val_loss: 0.6172
# Epoch 66/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6727 - loss: 0.6282 - val_accuracy: 0.6960 - val_loss: 0.6164
# Epoch 67/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6776 - loss: 0.6230 - val_accuracy: 0.6960 - val_loss: 0.6165
# Epoch 68/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6887 - loss: 0.6159 - val_accuracy: 0.6960 - val_loss: 0.6155
# Epoch 69/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6760 - loss: 0.6336 - val_accuracy: 0.6960 - val_loss: 0.6156
# Epoch 70/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7028 - loss: 0.6045 - val_accuracy: 0.7040 - val_loss: 0.6141
# Epoch 71/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6735 - loss: 0.6279 - val_accuracy: 0.7040 - val_loss: 0.6137
# Epoch 72/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6789 - loss: 0.6140 - val_accuracy: 0.7040 - val_loss: 0.6129
# Epoch 73/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6851 - loss: 0.6161 - val_accuracy: 0.7040 - val_loss: 0.6130
# Epoch 74/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6496 - loss: 0.6443 - val_accuracy: 0.7040 - val_loss: 0.6121
# Epoch 75/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7153 - loss: 0.5997 - val_accuracy: 0.7040 - val_loss: 0.6111
# Epoch 76/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7060 - loss: 0.6030 - val_accuracy: 0.7040 - val_loss: 0.6112
# Epoch 77/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6827 - loss: 0.6196 - val_accuracy: 0.7040 - val_loss: 0.6108
# Epoch 78/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6714 - loss: 0.6215 - val_accuracy: 0.7040 - val_loss: 0.6103
# Epoch 79/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6942 - loss: 0.6072 - val_accuracy: 0.7040 - val_loss: 0.6101
# Epoch 80/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6942 - loss: 0.6187 - val_accuracy: 0.7040 - val_loss: 0.6098
# Epoch 81/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6582 - loss: 0.6291 - val_accuracy: 0.7040 - val_loss: 0.6092
# Epoch 82/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6652 - loss: 0.6349 - val_accuracy: 0.7040 - val_loss: 0.6090
# Epoch 83/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6973 - loss: 0.6117 - val_accuracy: 0.7040 - val_loss: 0.6087
# Epoch 84/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6920 - loss: 0.6130 - val_accuracy: 0.7040 - val_loss: 0.6086
# Epoch 85/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6789 - loss: 0.6316 - val_accuracy: 0.7040 - val_loss: 0.6080
# Epoch 86/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6874 - loss: 0.6213 - val_accuracy: 0.7040 - val_loss: 0.6085
# Epoch 87/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6870 - loss: 0.6174 - val_accuracy: 0.7040 - val_loss: 0.6071
# Epoch 88/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6802 - loss: 0.6185 - val_accuracy: 0.7040 - val_loss: 0.6073
# Epoch 89/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6972 - loss: 0.6076 - val_accuracy: 0.7040 - val_loss: 0.6063
# Epoch 90/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6997 - loss: 0.6129 - val_accuracy: 0.7040 - val_loss: 0.6071
# Epoch 91/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6659 - loss: 0.6306 - val_accuracy: 0.7040 - val_loss: 0.6064
# Epoch 92/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6778 - loss: 0.6243 - val_accuracy: 0.7040 - val_loss: 0.6061
# Epoch 93/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6571 - loss: 0.6379 - val_accuracy: 0.7040 - val_loss: 0.6048
# Epoch 94/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6866 - loss: 0.6254 - val_accuracy: 0.7040 - val_loss: 0.6050
# Epoch 95/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6769 - loss: 0.6234 - val_accuracy: 0.7040 - val_loss: 0.6057
# Epoch 96/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.7250 - loss: 0.5957 - val_accuracy: 0.7040 - val_loss: 0.6054
# Epoch 97/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6721 - loss: 0.6326 - val_accuracy: 0.7040 - val_loss: 0.6048
# Epoch 98/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6850 - loss: 0.6185 - val_accuracy: 0.7040 - val_loss: 0.6046
# Epoch 99/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6749 - loss: 0.6215 - val_accuracy: 0.7040 - val_loss: 0.6046
# Epoch 100/100
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6938 - loss: 0.6104 - val_accuracy: 0.7040 - val_loss: 0.6045

# Final weights:
# [array([[ 0.01387656, -0.56391585, -0.6274722 , -0.9704919 , -0.23739512,
#          0.471359  , -0.14428882, -0.65321356],
#        [ 0.39380124,  0.64733493,  0.5115297 ,  0.22412714,  1.0213044 ,
#         -0.265583  , -0.44123992, -0.8425848 ]], dtype=float32), array([-0.24123664,  0.01396172, -0.04297299,  0.04417669, -0.12744541,
#        -0.137449  , -0.15751214,  0.01460731], dtype=float32)]
# [array([[ 0.4532187 ,  0.4205078 , -0.34335384, -0.54011226],
#        [-0.26819518, -0.43880692, -0.60991293,  0.15523522],
#        [ 0.01708853,  0.01677853, -0.65300184, -0.27251795],
#        [ 0.66300297, -0.82238275, -0.59643996,  0.26895237],
#        [-0.58829886,  0.06992258, -0.50701714, -0.25799555],
#        [-0.04194747,  0.06597867, -0.21286383, -0.746494  ],
#        [ 0.01116888,  0.21481156, -0.49692988, -0.14599936],
#        [ 0.74529725,  0.549079  , -0.689762  ,  0.43501124]],
#       dtype=float32), array([ 0.01043593, -0.14428535,  0.        , -0.02357376], dtype=float32)]
# [array([[ 0.89625585],
#        [ 0.8222473 ],
#        [-0.7936045 ],
#        [-0.13361979]], dtype=float32), array([-0.7570458], dtype=float32)]
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7410 - loss: 0.5733 

# Test accuracy: 0.7201
