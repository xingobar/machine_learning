import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

## fix random seed for reproducibility
np.random.seed(42)

## load dataset
df = pd.read_csv('~/Downloads/international-airline-passengers.csv',
				 engine='python',usecols=[1],skipfooter=3)
df = df.values
df = df.astype('float32')

## normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

## split into train and test sets
train_size = df.shape[0] * 0.7
test_size =df.shape[0] - train_size
train,test  = df[:train_size,:] , df[train_size:df.shape[0],:]
print('Train Shape : {}'.format(train.shape))
print('Test Shape : {}'.format(test.shape))


### x is the number of passengers at a given time t 
### y is the number of passengers at the next time (t+1)

def create_dataset(dataset,look_back=1):
	X,y = [],[]
	for i in range(len(dataset) - look_back -1):
		a = dataset[i:(i+look_back),0]
		X.append(a)
		y.append(dataset[i+look_back,0])
	return np.array(X),np.array(y)


x_train,y_train = create_dataset(train)
x_test,y_test = create_dataset(test)
## reshape input to be [samples,timesteps,features]
look_back=1
x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))


## create and fit the LSTM network
model = Sequential()
model.add(LSTM(4,input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,nb_epoch=100,batch_size=1,verbose=2)



## make predictions
trainPrediction = model.predict(x_train)
testPrediction = model.predict(x_test)

## invert predictions
## ensure that performance is reported in same units as the original data
trainPrediction = scaler.inverse_transform(trainPrediction)
y_train = scaler.inverse_transform(y_train)
testPrediction = scaler.inverse_transform(testPrediction)
y_test = scaler.inverse_transform(y_test)

## calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train,trainPrediction[:,0]))
testScore = math.sqrt(mean_squared_error(y_test,testPrediction[:,0]))
print('Train Score of RMSE: {}'.format(trainScore))
print('Test Score of RMSE :{}'.format(testScore))























