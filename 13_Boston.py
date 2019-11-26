from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
print(boston.keys())
dfX = pd.DataFrame(boston.data, columns=boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=["MEDV"])
df = pd.concat([dfX, dfy], axis=1)

x = boston.data
y = boston.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
dataset = df.values
model = Sequential()
model.add(Dense(30, input_dim=13, activation='sigmoid'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=400, batch_size=20)

y_prediction = model.predict(x_test).flatten()
for i in range(20):
    label = y_test[i]
    prediction = y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))

plt.scatter(y_test, y_prediction)
plt.xlabel("y_test")
plt.ylabel("y_prediction")
plt.show()
