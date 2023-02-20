# M. Alwarawrah
import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import seed
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import (losses, metrics) 
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# start recording time
t_initial = time.time()

#normalization function
def normalize(features):
    norm = (features - features.mean()) / features.std()
    return norm

#accuracy function
def accu_func(pred, y_test):
    y_actual = pred[:,0]
    error = ((y_actual-y_test)/y_test)
    percent_error = error.mean()
    accuracy = 100 - percent_error
    return percent_error, accuracy

# define regression model
def regression_model(n_cols):
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#plot Loss and Accuracy vs epoch
def plot_loss(train_loss, val_loss):
    plt.clf()
    plt.plot(train_loss, color='k', label = 'Training Loss')
    plt.plot(val_loss, color='r', label = 'Validation Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=16)
    plt.savefig('loss_vs_epoch.png')

# Download data from: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv
concrete_data = pd.read_csv('concrete_data.csv')

#create a file to write the outputs
output_file = open('output.txt','w')

#print
print(concrete_data.head(), file=output_file)
print('data domensions: {}'.format(concrete_data.shape), file=output_file)
print(concrete_data.describe(), file=output_file)

#check if data has null
print('number of null in data:', file=output_file)
print(concrete_data.isnull().sum(), file=output_file)

#define features and target data
features = concrete_data.drop(['Strength'],axis=1)
target = concrete_data['Strength']

#print
print('Features:', file=output_file)
print(features.head(), file=output_file)
print('target:', file=output_file)
print(target.head(), file=output_file)

#normalize features
features_norm = normalize(features)
print('Features after normalization:', file=output_file)
print(features_norm.head(), file=output_file)

#split the dataset to train and test
x_train, x_test, y_train, y_test = train_test_split( features_norm, target, test_size=0.3, random_state=4)

#find the number of columns in x_train
n_cols = x_train.shape[1]

#build regression model
model = regression_model(n_cols)

# fit the model
epochs = 300
results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=2)

train_loss = results.history['loss']
val_loss = results.history['val_loss']
  
# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)

#prediction
predictions = model.predict(x_test)

#find accuracy and percent error
percent_error, accuracy = accu_func(predictions, y_test)

print("Training loss: %5.2f at last epoch: %d"%(train_loss[-1],epochs), file=output_file)
print("Training loss: %5.2f at last epoch: %d"%(train_loss[-1],epochs))
print("Validation loss: %5.2f at last epoch: %d"%(val_loss[-1],epochs), file=output_file)
print("Validation loss: %5.2f at last epoch: %d"%(val_loss[-1],epochs))
print("Validation Accuracy: %5.3f & Validation Percent Error: %5.3f"%(accuracy,percent_error), file=output_file)
print("Validation Accuracy: %5.3f & Validation Percent Error: %5.3f"%(accuracy,percent_error))

#plot train and validation loss
plot_loss(train_loss, val_loss)

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))