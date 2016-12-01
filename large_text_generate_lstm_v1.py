import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from kreas.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = '/Users/xingobar/Desktop/machine-learning/data/text/wonderland.txt'
raw_text = open(filename,'r').read()
raw_text = raw_text.lower()

## creating mapping of character to integer
chars = sorted(list(set(raw_text)))
char_to_int = dict((char,idx) for idx,char in enumerate(chars))

n_chars = len(raw_text)
n_vocabulary = len(chars)
print('Total character : {}'.format(n_chars))
print('Total Vocabulary : {}'.format(n_vocabulary))

## prepare the dataset of input to output pairs encoded as integer
sequence = 100
X=[]
y=[]
for i in range(0,n_chars - sequence,1):
	seq_in = raw_text[i:i+sequence]
	seq_out = raw_text[i+sequence]
	X.append([char_to_int[seq] for seq in seq_in])
	y.append(char_to_int[seq_out])

n_pattern = len(X)
print('Number of pattern : {}'.format(n_pattern))

## (sample,timestep,feature)
X = np.reshape(X,(n_pattern,sequence,1))
## normalize
X = X / float(n_vocabulary)
## one-hot encode
y = np_utils.to_category(y)
print('X Shape : {]'.format(X.shape))
print('y shape : {}'.format(y.shape))


## define lstm model
model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')


filepath = 'big_weight_improvement:{epoch:02d}-{loss:04f}.hd5'
callbacks = ModelCheckpoint(filepath,monitor='loss',save_best_only=True,mode='min',verbose=1)
model.fit(X,y,nb_epoch=50,batch_size=64,callbacks=[callbacks])












