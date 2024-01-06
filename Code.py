import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Loading the Training Dataset.

dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
dataset_train.describe()

dataset_train.head()

training_set=dataset_train.iloc[:,1:2].values
print(training_set.shape)

plt.figure(figsize=(10,5))
plt.plot(training_set, color ='green');
plt.ylabel('Stock Price')
plt.title('Google Stock Price')
plt.xlabel('Time')
plt.show()

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler(feature_range=(0,1))
scaled_training_set=scalar.fit_transform(training_set)
scaled_training_set

plt.figure(figsize=(10,5))
plt.plot(scaled_training_set);
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show()

Creating X_train and y_train Data Structures.

X_train=scaled_training_set[59:1257]
y_train=scaled_training_set[60:1258]
    
X_train=np.array(X_train)
y_train=np.array(y_train)
print(X_train.shape)
print(y_train.shape)

Reshape the Data.

X_train=np.reshape(X_train,(1198,1,1))
print(X_train.shape)

import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

Adding First 50 units.

regressor=Sequential
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

Adding Second 50 units.

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

Adding Third 50 units.

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

Adding Fourth 50 units.

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)
