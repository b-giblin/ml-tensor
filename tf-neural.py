import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('data.csv')

x = df.drop('date', axis=1).values
y = df['rate'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential([
  keras.layers.Dense(64, activation='relu',input_shape=(x_train.shape[1],)),
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(1) # no activation as regression output is continuous
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=32)

loss, mae = model.evaluate(x_test, y_test)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae))
predictions = model.predict(x_test)