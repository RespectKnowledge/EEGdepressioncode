# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 07:26:37 2019

@author: Abdul Qayyum
"""

#%% combination of CNN and LSTM using dpression dataset for EC dataset
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Activation, Dropout, GRU
import pandas as pd
from keras.optimizers import SGD
import math
from keras.layers import Conv1D,MaxPool1D,LSTM, Dropout

from keras.models import Sequential
from keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense

# Control Dataset
import os
import scipy
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from scipy import io, signal
import matplotlib.pyplot as plt
import dtcwt
import numpy as np
import itertools
import pywt
#from __future__ import print_function
from matplotlib import pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
################## Normalize dataset into zero and one for all features
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#X_short = scaler.fit_transform(X_short) X_short is  a data matrix
def Normaliz_f(X1):
    min_max_scaler = preprocessing.MinMaxScaler()
    X1 = min_max_scaler.fit_transform(X1)
    return X1

def Normaliz_f1(X11):
    min_max_scaler = preprocessing.MaxAbsScaler()
    X11 = min_max_scaler.fit_transform(X11)
    return X11
################## Normalize dataset into zero and one for all features
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#X_short = scaler.fit_transform(X_short) X_short is  a data matrix

# load dataset
dataDir = "/raid/Home/Users/aqayyum/pymultimodel/EEGmodels/controls/"
#dir_seg = dataDir + "/Ahmad/"
#dir_segEO=os.listdir(dir_seg)[0]
#dir_segEC=os.listdir(dir_seg)[1]
# function for normalization feature matrix
#def numfun(f):
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    scaledtrain = scaler.fit_transform(f)
#    return scaledtrain
#Predicted = dir_data + "/preds_8300128res/"
matsEOControl = []
matsECControl = []
dataec=[]
for file in os.listdir( dataDir ) :
    EO=os.listdir(dataDir+file)[0]
    EC=os.listdir(dataDir+file)[1]
    matsEOControl.append( scipy.io.loadmat(os.path.join(dataDir+file,EO )))
    matsECControl.append(scipy.io.loadmat(os.path.join(dataDir+file,EC ) ))

# make time windows of  EC dataset
from numpy import array
n=15000
samples = list()
length = 256
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecControlEC=[]
y=[]
#dataee=[]
#dataee1=[]
for dd in matsECControl:
    vv=dd['data']
    vv1=vv.transpose()
#    vv11=Normaliz_f(vv1)
#    vv2=Normaliz_f1(vv1)
    #vv1=vv1[:,1:11]
    samples1=conactfun(vv1)
    dataControlEC = array(samples1)
    dataecControlEC.append(dataControlEC) 
    y.append(1)
#    dataee.append(vv1)
#    dataee1.append(vv2)
  
# make time windows of  EO dataset 
from numpy import array
n=15000
samples = list()
#length = 500
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecControlEO=[]
y=[]
for dd in matsEOControl :
    vv=dd['data']
    vv1=vv.transpose()
    #vv1=vv1[:,1:11]
#    vv3=Normaliz_f1(vv1)
    samples11=conactfun(vv1)
    dataControlEO = array(samples11)
    dataecControlEO.append(dataControlEO) 
    y.append(1)    
    
    
    
#datamain=array(dataec) 
#label create for control class dataset   
y11=y*59   

label=np.array(y11)
#% make class2 MDD dataset class
import os
import scipy
from scipy import io, signal
import matplotlib.pyplot as plt
import dtcwt
import numpy as np
import itertools
import pywt

dataDir = "/raid/Home/Users/aqayyum/pymultimodel/EEGmodels/MDD/"
#dir_segEO=os.listdir(dir_seg)[0]
#dir_segEC=os.listdir(dir_seg)[1]

#Predicted = dir_data + "/preds_8300128res/"
matsEOMDD = []
matsECMDD = []
dataec=[]

for file in os.listdir( dataDir ) :
    EO=os.listdir(dataDir+file)[0]
    EC=os.listdir(dataDir+file)[1]
    matsEOMDD.append( scipy.io.loadmat(os.path.join(dataDir+file,EO )))
    matsECMDD.append(scipy.io.loadmat(os.path.join(dataDir+file,EC ) ))
    
    
    
# make window for MMD class for EC dataset
from numpy import array
n=15000
samples = list()
#length = 500
y=[]
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecMDDEC=[]
for dd in matsECMDD:
    vv=dd['data']
    vv1=vv.transpose()
    #vv1=vv1[:,1:11]
#    vv4=Normaliz_f1(vv1)
    samples4=conactfun(vv1)
    dataMDDEC = array(samples4)
    dataecMDDEC.append(dataMDDEC)
    y.append(0)

################################Data preprocessing######################3

# Data for EO for MDD class
  
from numpy import array
n=15000
samples = list()
#length = 500
y=[]
def conactfun(vv1):
    # step over the 5,000 in jumps of 200
    for i in range(0,n,length):
        # grab from i to i + 200
        sample = vv1[i:i+length,:]
        samples.append(sample)
        print(len(samples))
    return(samples)
dataecMDDEO=[]
for dd in matsEOMDD:
    vv=dd['data']
    vv1=vv.transpose()
#    vv5=Normaliz_f1(vv1)
    #vv1=vv1[:,1:11]
    samples5=conactfun(vv1)
    dataMDDEO = array(samples5)
    dataecMDDEO.append(dataMDDEO)
    y.append(0)   

dataControlEC.shape[0]
# create labels for MDD class dataset
y22=y*59
y22=np.array(y22)
# For control dataset create labels
yoneECC=np.array(np.ones(dataControlEC.shape[0]))
yoneEOC=np.array(np.ones(dataControlEO.shape[0]))
yzeroECC=np.array(np.zeros(dataControlEC.shape[0]))
yzeroEOC=np.array(np.zeros(dataControlEO.shape[0]))
# for MDD dataset create labels
# For control dataset create labels
yoneECM=np.array(np.ones(dataMDDEC.shape[0]))
yoneEOM=np.array(np.ones(dataMDDEO.shape[0]))
yzeroECM=np.array(np.zeros(dataMDDEC.shape[0]))
yzeroEOM=np.array(np.zeros(dataMDDEO.shape[0]))

# concatenate Labels for both classes label for control class, y22 for MMD class
labelsEC=np.concatenate((yoneECC,yzeroECM),axis=0)
# concatenate dataset for EC for two classes(control and MMD).   
DatamatrixEC=np.concatenate((dataControlEO,dataMDDEO),axis=0)
X_shortEC=DatamatrixEC
y_shortEC=labelsEC

# concatenate for Eye close dataset for two classes
labelsEO=np.concatenate((yoneEOC,yzeroEOM),axis=0)
# concatenate dataset for EC for two classes(control and MMD).   
DatamatrixEO=np.concatenate((dataControlEC,dataMDDEC),axis=0)
X_shortEO=DatamatrixEO
y_shortEO=labelsEO

## define labels used in this dataset
#target = [[i for i in range(1,3717)]]
#target = np.array(target, dtype=float)
    
    
##datamain=array(dataec)   
#def create_dataset(dataset, look_back=1):
#    dataX, dataY = [], []
#    for i in range(len(dataset)-look_back):
#        dataX.append(dataset[i:(i+look_back), 0])
#        dataY.append(dataset[i + look_back, 0])
#    return np.array(dataX), np.array(dataY)
#
#def one_hot_events(events):
#    # helper function for one-hot encoding the events
#    events_list = list(events)
#    lb = preprocessing.LabelBinarizer()
#    lb.fit(events_list)
#    events_1hot = lb.transform(events_list)
#    return events_1hot, lb

# create for eye open
import random
seed=42
random.seed(seed)
# shuffled dataset for training and testing
from sklearn.model_selection import StratifiedShuffleSplit

# use strat. shuffle split to get indices for test and training data 
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
sss.get_n_splits(X_shortEO, y_shortEO)

# create train and test dataset for classification
# take the indices generated by stratified shuffle split and make the test and training datasets
for train_index, test_index in sss.split(X_shortEO, y_shortEO):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_trainEO, X_testEO = X_shortEO[train_index], X_shortEO[test_index]
    y_trainEO, y_testEO = y_shortEO[train_index], y_shortEO[test_index]
  
  
#X_trainEO = X_trainEO.astype('float32')
#X_testEO = X_testEO.astype('float32')
#X_trainEO /= 255
#X_testEO /= 255    
y_train_hotEO = np_utils.to_categorical(y_trainEO, 2)
y_test_hotEO=np_utils.to_categorical(y_testEO, 2)

#%
# create for EC

import random
seed=42
random.seed(seed)
# shuffled dataset for training and testing
from sklearn.model_selection import StratifiedShuffleSplit

# use strat. shuffle split to get indices for test and training data 
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
sss.get_n_splits(X_shortEC, y_shortEC)

# create train and test dataset for classification
# take the indices generated by stratified shuffle split and make the test and training datasets
for train_index, test_index in sss.split(X_shortEC, y_shortEC):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_trainEC, X_testEC = X_shortEC[train_index], X_shortEC[test_index]
    y_trainEC, y_testEC = y_shortEC[train_index], y_shortEC[test_index]
    
y_train_hotEC = np_utils.to_categorical(y_trainEC, 2)
y_test_hotEC=np_utils.to_categorical(y_testEC, 2)
#X_trainEC = X_trainEC.astype('float32')
#X_testEC = X_testEC.astype('float32')
#X_trainEC /= 255
#X_testEC /= 255  
#https://scikit-learn.org/stable/modules/preprocessing.html
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#X_trainEC = min_max_scaler.fit_transform(X_trainEC)
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#X_testEC = min_max_scaler.fit_transform(X_testEC)

temporal_dimension = X_trainEC.shape[1]
num_channels = X_trainEC.shape[2]
num_classes = 2
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
rms=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adagrad=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adadelt=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adamx=optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#########################################EO######################################
#%% LSTM+CNN for eye open dataset
#temporal_dimension = X_trainEO.shape[1]
#num_channels = X_trainEO.shape[2]
#num_classes = 2
#modellstmEC = Sequential()
#modellstmEC.add(Conv1D(64, 5, activation='elu', input_shape=(temporal_dimension,num_channels)))
#modellstmEC.add(MaxPool1D(3,1,))
#modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
##modellstmEC.add(Conv1D(48, 5, activation='elu'))
##modellstmEC.add(MaxPool1D(3,1,))
##modellstmEC.add(Dropout(0.2))
##    # model.add(BatchNormalization())
##modellstmEC.add(Conv1D(24, 3, activation='elu'))
##modellstmEC.add(MaxPool1D(3,1,))
##modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstmEC.add(LSTM(64, return_sequences=True))
#modellstmEC.add(LSTM(128)) #stacked recurrent layers said to enable deeper time series learningmodel.add(Dense(bdfData.info['nchan']))
##model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
#modellstmEC.add(Dense(units=2))
#modellstmEC.add(Activation('sigmoid'))
#modellstmEC.summary()
#
## Compiling the RNN
##modelGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='binary_crossentropy',metrics=['accuracy'])
## Fitting to the training set
##history=modelGRU.fit(X_train,y_train,epochs=50,batch_size=150,validation_data=(X_test, tesy), verbose=2, shuffle=False)
##callbacks_list = [
##    keras.callbacks.ModelCheckpoint(
##        filepath='best_modellstmmulti.{epoch:02d}-{val_loss:.2f}.h5',
##        monitor='val_loss', save_best_only=True),
##    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
##]
#
#modellstmEC.compile(loss='binary_crossentropy',
#                optimizer=adamx, metrics=['accuracy'])
#
#BATCH_SIZE = 100
#EPOCHS = 50
#
#history = modellstmEC.fit(X_trainEO,
#                      y_train_hotEO,
#                      batch_size=BATCH_SIZE,
#                      epochs=EPOCHS,
#                      validation_split=0.2,
#                      verbose=1)
#
##save.modellstmEC('modellstmEC.h5')
#
##import matplotlib.pyplot as plt
##plt.figure(figsize=(6, 4))
##loss = history.history['loss']
##val_loss = history.history['val_loss']
##epochs = range(1, len(loss) + 1)
##plt.plot(epochs, loss, color='red', label='Training loss')
##plt.plot(epochs, val_loss, color='green', label='Validation loss')
##plt.title('Training and validation loss')
##plt.xlabel('Epochs',fontsize=10)
##plt.ylabel('Loss',fontsize=10)
##plt.legend()
##plt.savefig("Losseseclstmc")
##plt.show()
##
###plotting training and validation accuracy
##plt.figure(figsize=(6, 4))
##acc = history.history['acc']
##val_acc = history.history['val_acc']
##plt.plot(epochs, acc, color='red', label='Training acc')
##plt.plot(epochs, val_acc, color='green', label='Validation acc')
##plt.title('Training and validation accuracy')
##plt.xlabel('Epochs',fontsize=10)
##plt.ylabel('Accuracy',fontsize=10)
##plt.legend()
##plt.savefig("accuracyeclstmc")
##plt.show()
##%accuracy on test data
##y_testEC = np_utils.to_categorical(X_trainEO,2)
#score = modellstmEC.evaluate(X_trainEO, y_train_hotEO, verbose=1)
#
#print('\nAccuracy on train data: %0.2f' % score[1])
#print('\nLoss on train data: %0.2f' % score[0])
#
##y_testEC = np_utils.to_categorical(X_trainEO,2)
#y_test_hotEO=np_utils.to_categorical(y_testEO, 2)
#score = modellstmEC.evaluate(X_testEO, y_test_hotEO, verbose=1)
#
#print('\nAccuracy on test data: %0.2f' % score[1])
#print('\nLoss on test data: %0.2f' % score[0])
##Accuracy on test data: 0.95
##%##################        Print confusion matrix for training data and testing datasets
##LABELS = ['Control','MDD']
##def show_confusion_matrix(validations, predictions):
##
##    matrix = metrics.confusion_matrix(validations, predictions)
##    plt.figure(figsize=(2, 2))
##    sns.heatmap(matrix,
##                cmap='coolwarm',
##                linecolor='white',
##                linewidths=1,
##                xticklabels=LABELS,
##                yticklabels=LABELS,
##                annot=True,
##                fmt='d')
##    plt.title('Confusion Matrix')
##    plt.ylabel('True Label')
##    plt.xlabel('Predicted Label')
##    plt.show()
#
#
#
#y_pred_train = modellstmEC.predict(X_trainEO)
## Take the class with the highest probability from the train predictions
#max_y_pred_train = np.argmax(y_pred_train, axis=1)
#y_train_hotEO1=np.argmax(y_train_hotEO, axis=1)
#print(classification_report(y_train_hotEO1, max_y_pred_train))
#
#y_pred_test = modellstmEC.predict(X_testEO)
## Take the class with the highest probability from the test predictions
#max_y_pred_test = np.argmax(y_pred_test, axis=1)
#max_y_test = np.argmax(y_test_hotEO, axis=1)
#print(classification_report(max_y_test, max_y_pred_test))
##show_confusion_matrix(max_y_test, max_y_pred_test)
#modellstmEC.save("modelCNNLSTMEO.h5")



#%%############################### LSTMCNN architecture####################################################3
# Defined model architecture
#modellstm = Sequential()
#modellstm.add(Conv1D(64, 5, activation='elu', input_shape=(temporal_dimension,num_channels)))
#modellstm.add(MaxPool1D(3,1,))
#modellstm.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstm.add(Conv1D(48, 5, activation='elu'))
#modellstm.add(MaxPool1D(3,1,))
#modellstm.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstm.add(Conv1D(24, 3, activation='elu'))
#modellstm.add(MaxPool1D(3,1,))
#modellstm.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstm.add(LSTM(128, return_sequences=True))
#modellstm.add(LSTM(64)) #stacked recurrent layers said to enable deeper time series learningmodel.add(Dense(bdfData.info['nchan']))
##model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
#modellstm.add(Dense(units=2))
#modellstm.add(Activation('sigmoid'))
#modellstm.summary()
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Reshape
#from keras.layers import Conv2D, MaxPooling2D
#from keras.utils import np_utils
#from keras.layers import Dense, Dropout
#from keras.layers import LSTM
#from keras.models import Sequential
#from keras.layers import Dense
#from sklearn.model_selection import StratifiedKFold
#import numpy
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#####################################cross validation
#cvscores = []
#for train, test in kfold.split(X_trainEC, y_trainEC):
#    temporal_dimension = X_trainEC.shape[1]
#    num_channels = X_trainEC.shape[2]
#    num_classes = 2
#    modellstm = Sequential()
#    modellstm.add(Conv1D(64, 5, activation='elu', input_shape=(temporal_dimension,num_channels)))
#    modellstm.add(MaxPool1D(3,1,))
#    modellstm.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#    modellstm.add(Conv1D(48, 5, activation='elu'))
#    modellstm.add(MaxPool1D(3,1,))
#    modellstm.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#    modellstm.add(Conv1D(24, 3, activation='elu'))
#    modellstm.add(MaxPool1D(3,1,))
#    modellstm.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#    modellstm.add(LSTM(128, return_sequences=True))
#    modellstm.add(LSTM(64)) #stacked recurrent layers said to enable deeper time series learningmodel.add(Dense(bdfData.info['nchan']))
#    modellstm.add(Dense(units=1))
#    modellstm.add(Activation('sigmoid'))
#    modellstm.summary()
#    modellstm.compile(loss='binary_crossentropy',
#                optimizer='adam', metrics=['accuracy'])
#    callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#        monitor='val_loss', save_best_only=True),keras.callbacks.EarlyStopping(monitor='acc', patience=1)]
#    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#    BATCH_SIZE = 100
#    EPOCHS = 50
#    history = modellstm.fit(X_trainEC[train],
#                      y_trainEC[train],
#                      batch_size=BATCH_SIZE,
#                      epochs=EPOCHS,
#                      callbacks=callbacks_list,
#                      validation_split=0.1,
#                      verbose=1)
#    plt.figure(figsize=(6, 4))
#    plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
#    plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
#    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
#    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
#    plt.title('Model Accuracy and Loss')
#    plt.ylabel('Accuracy and Loss')
#    plt.xlabel('Training Epoch')
#    plt.ylim(0)
#    plt.legend()
#    plt.show()
#    scores = modellstm.evaluate(X_trainEC[test], y_trainEC[test], verbose=0)
#    print("%s: %.2f%%" % (modellstm.metrics_names[1], scores[1]*100))
#    cvscores.append(scores[1] * 100)
#    
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#
#np.savetxt('cvscoreseccnn.csv', cvscores, delimiter=',', fmt='%.5f')

#%% LSTM+CNN for eye close dataset
#temporal_dimension = X_trainEC.shape[1]
#num_channels = X_trainEC.shape[2]
#num_classes = 2
#modellstmEC = Sequential()
#modellstmEC.add(Conv1D(64, 5, activation='elu', input_shape=(temporal_dimension,num_channels)))
#modellstmEC.add(MaxPool1D(3,1,))
#modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstmEC.add(Conv1D(48, 5, activation='elu'))
#modellstmEC.add(MaxPool1D(3,1,))
#modellstmEC.add(Dropout(0.2))
##    # model.add(BatchNormalization())
##modellstmEC.add(Conv1D(24, 3, activation='elu'))
##modellstmEC.add(MaxPool1D(3,1,))
##modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstmEC.add(LSTM(128, return_sequences=True))
#modellstmEC.add(LSTM(128)) #stacked recurrent layers said to enable deeper time series learningmodel.add(Dense(bdfData.info['nchan']))
##model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
#modellstmEC.add(Dense(units=2))
#modellstmEC.add(Activation('sigmoid'))
#modellstmEC.summary()
#
## Compiling the RNN
##modelGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='binary_crossentropy',metrics=['accuracy'])
## Fitting to the training set
##history=modelGRU.fit(X_train,y_train,epochs=50,batch_size=150,validation_data=(X_test, tesy), verbose=2, shuffle=False)
##callbacks_list = [
##    keras.callbacks.ModelCheckpoint(
##        filepath='best_modellstmmulti.{epoch:02d}-{val_loss:.2f}.h5',
##        monitor='val_loss', save_best_only=True),
##    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
##]
#
#modellstmEC.compile(loss='binary_crossentropy',
#                optimizer='adam', metrics=['accuracy'])
#
#BATCH_SIZE = 100
#EPOCHS = 100
#
#history = modellstmEC.fit(X_trainEC,
#                      y_train_hotEC,
#                      batch_size=BATCH_SIZE,
#                      epochs=EPOCHS,
#                      validation_split=0.2,
#                      verbose=1)
#
##save.modellstmEC('modellstmEC.h5')
#
##import matplotlib.pyplot as plt
##plt.figure(figsize=(6, 4))
##loss = history.history['loss']
##val_loss = history.history['val_loss']
##epochs = range(1, len(loss) + 1)
##plt.plot(epochs, loss, color='red', label='Training loss')
##plt.plot(epochs, val_loss, color='green', label='Validation loss')
##plt.title('Training and validation loss')
##plt.xlabel('Epochs',fontsize=10)
##plt.ylabel('Loss',fontsize=10)
##plt.legend()
##plt.savefig("Losseseclstmc")
##plt.show()
##
###plotting training and validation accuracy
##plt.figure(figsize=(6, 4))
##acc = history.history['acc']
##val_acc = history.history['val_acc']
##plt.plot(epochs, acc, color='red', label='Training acc')
##plt.plot(epochs, val_acc, color='green', label='Validation acc')
##plt.title('Training and validation accuracy')
##plt.xlabel('Epochs',fontsize=10)
##plt.ylabel('Accuracy',fontsize=10)
##plt.legend()
##plt.savefig("accuracyeclstmc")
##plt.show()
##%accuracy on test data
#y_testEC = np_utils.to_categorical(y_testEC,2)
#
#score = modellstmEC.evaluate(X_testEC, y_testEC, verbose=1)
#
#print('\nAccuracy on test data: %0.2f' % score[1])
#print('\nLoss on test data: %0.2f' % score[0])
##Accuracy on test data: 0.95
##%##################        Print confusion matrix for training data and testing datasets
##LABELS = ['Control','MDD']
##def show_confusion_matrix(validations, predictions):
##
##    matrix = metrics.confusion_matrix(validations, predictions)
##    plt.figure(figsize=(2, 2))
##    sns.heatmap(matrix,
##                cmap='coolwarm',
##                linecolor='white',
##                linewidths=1,
##                xticklabels=LABELS,
##                yticklabels=LABELS,
##                annot=True,
##                fmt='d')
##    plt.title('Confusion Matrix')
##    plt.ylabel('True Label')
##    plt.xlabel('Predicted Label')
##    plt.show()
#
#
#
#y_pred_train = modellstmEC.predict(X_trainEC)
## Take the class with the highest probability from the train predictions
#max_y_pred_train = np.argmax(y_pred_train, axis=1)
#print(classification_report(y_trainEC, max_y_pred_train))
#
#
#y_pred_test = modellstmEC.predict(X_testEC)
## Take the class with the highest probability from the test predictions
#max_y_pred_test = np.argmax(y_pred_test, axis=1)
#max_y_test = np.argmax(y_testEC, axis=1)
#print(classification_report(max_y_test, max_y_pred_test))
##show_confusion_matrix(max_y_test, max_y_pred_test)
#modellstmEC.save("modelCNNLSTMEC.h5")
##print(classification_report(max_y_test, max_y_pred_test))
##
##modellstmEC.summary()
##from keras import backend as K
##for l in range(len(modellstmEC.layers)):
##    print(l, modellstmEC.layers[l])
##
##getFeature = K.function([modellstmEC.layers[0].input, K.learning_phase()],
##                        [modellstmEC.layers[5].output])
##
##exTrain3000 = getFeature([X_trainEC, 0])[0]
##print(exTrain3000.shape)
##print(y_train_hotEC.shape)
##y_train_hotEC=y_train_hotEC
##from sklearn.model_selection import train_test_split
##X_traindd, X_testdd, y_traindd, y_testdd = train_test_split(exTrain3000, y_train_hotEC, test_size=0.2, stratify=y_train_hotEC)
##from sklearn.neighbors import KNeighborsClassifier
##from sklearn.model_selection import GridSearchCV
##from sklearn.multiclass import OneVsRestClassifier
##from sklearn.neighbors import KNeighborsClassifier
##from sklearn.model_selection import GridSearchCV
##
##parameters = {"n_neighbors": [1, 5, 10, 30],
##              "weights": ['uniform', 'distance'],
##              "metric": ['minkowski','euclidean','manhattan'],
##              "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}
##kclf = KNeighborsClassifier()
##kgclf = GridSearchCV(kclf, param_grid=parameters)
##
##kgclf.fit(X_traindd, y_traindd)





#######################################1DCNN+GRU+EO###################################
#from numpy import concatenate
#from matplotlib import pyplot
#from pandas import read_csv
#from pandas import DataFrame
#from pandas import concat
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM, Activation, Dropout, GRU
#import pandas as pd
#from keras.optimizers import SGD
#import math
##%% LSTM+CNN for eye open dataset
#temporal_dimension = X_trainEO.shape[1]
#num_channels = X_trainEO.shape[2]
#num_classes = 2
#modellstmEC = Sequential()
#modellstmEC.add(Conv1D(64, 5, activation='elu', input_shape=(temporal_dimension,num_channels)))
#modellstmEC.add(MaxPool1D(3,1,))
#modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstmEC.add(Conv1D(48, 5, activation='elu'))
#modellstmEC.add(MaxPool1D(3,1,))
#modellstmEC.add(Dropout(0.2))
##    # model.add(BatchNormalization())
##modellstmEC.add(Conv1D(24, 3, activation='elu'))
##modellstmEC.add(MaxPool1D(3,1,))
##modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstmEC.add(GRU(units=50, return_sequences=True,activation='tanh'))
#modellstmEC.add(GRU(units=128, activation='tanh'))
#modellstmEC.add(Dropout(0.2))
##modellstmEC.add(LSTM(128)) #stacked recurrent layers said to enable deeper time series learningmodel.add(Dense(bdfData.info['nchan']))
##model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
#modellstmEC.add(Dense(units=2))
#modellstmEC.add(Activation('sigmoid'))
#modellstmEC.summary()
#
## Compiling the RNN
##modelGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='binary_crossentropy',metrics=['accuracy'])
## Fitting to the training set
##history=modelGRU.fit(X_train,y_train,epochs=50,batch_size=150,validation_data=(X_test, tesy), verbose=2, shuffle=False)
##callbacks_list = [
##    keras.callbacks.ModelCheckpoint(
##        filepath='best_modellstmmulti.{epoch:02d}-{val_loss:.2f}.h5',
##        monitor='val_loss', save_best_only=True),
##    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
##]
#
#modellstmEC.compile(loss='binary_crossentropy',
#                optimizer=adamx, metrics=['accuracy'])
#
#BATCH_SIZE = 100
#EPOCHS = 50
#
#history = modellstmEC.fit(X_trainEO,
#                      y_train_hotEO,
#                      batch_size=BATCH_SIZE,
#                      epochs=EPOCHS,
#                      validation_split=0.2,
#                      verbose=1)
#
##save.modellstmEC('modellstmEC.h5')
#
##import matplotlib.pyplot as plt
##plt.figure(figsize=(6, 4))
##loss = history.history['loss']
##val_loss = history.history['val_loss']
##epochs = range(1, len(loss) + 1)
##plt.plot(epochs, loss, color='red', label='Training loss')
##plt.plot(epochs, val_loss, color='green', label='Validation loss')
##plt.title('Training and validation loss')
##plt.xlabel('Epochs',fontsize=10)
##plt.ylabel('Loss',fontsize=10)
##plt.legend()
##plt.savefig("Losseseclstmc")
##plt.show()
##
###plotting training and validation accuracy
##plt.figure(figsize=(6, 4))
##acc = history.history['acc']
##val_acc = history.history['val_acc']
##plt.plot(epochs, acc, color='red', label='Training acc')
##plt.plot(epochs, val_acc, color='green', label='Validation acc')
##plt.title('Training and validation accuracy')
##plt.xlabel('Epochs',fontsize=10)
##plt.ylabel('Accuracy',fontsize=10)
##plt.legend()
##plt.savefig("accuracyeclstmc")
##plt.show()
##%accuracy on test data
##y_testEC = np_utils.to_categorical(X_trainEO,2)
#score = modellstmEC.evaluate(X_trainEO, y_train_hotEO, verbose=1)
#
#print('\nAccuracy on train data: %0.2f' % score[1])
#print('\nLoss on train data: %0.2f' % score[0])
#
##y_testEC = np_utils.to_categorical(X_trainEO,2)
#y_test_hotEO=np_utils.to_categorical(y_testEO, 2)
#score = modellstmEC.evaluate(X_testEO, y_test_hotEO, verbose=1)
#
#print('\nAccuracy on test data: %0.2f' % score[1])
#print('\nLoss on test data: %0.2f' % score[0])
##Accuracy on test data: 0.95
##%##################        Print confusion matrix for training data and testing datasets
##LABELS = ['Control','MDD']
##def show_confusion_matrix(validations, predictions):
##
##    matrix = metrics.confusion_matrix(validations, predictions)
##    plt.figure(figsize=(2, 2))
##    sns.heatmap(matrix,
##                cmap='coolwarm',
##                linecolor='white',
##                linewidths=1,
##                xticklabels=LABELS,
##                yticklabels=LABELS,
##                annot=True,
##                fmt='d')
##    plt.title('Confusion Matrix')
##    plt.ylabel('True Label')
##    plt.xlabel('Predicted Label')
##    plt.show()
#
#
#
#y_pred_train = modellstmEC.predict(X_trainEO)
## Take the class with the highest probability from the train predictions
#max_y_pred_train = np.argmax(y_pred_train, axis=1)
#y_train_hotEO1=np.argmax(y_train_hotEO, axis=1)
#print(classification_report(y_train_hotEO1, max_y_pred_train))
#
#y_pred_test = modellstmEC.predict(X_testEO)
## Take the class with the highest probability from the test predictions
#max_y_pred_test = np.argmax(y_pred_test, axis=1)
#max_y_test = np.argmax(y_test_hotEO, axis=1)
#print(classification_report(max_y_test, max_y_pred_test))
##show_confusion_matrix(max_y_test, max_y_pred_test)
#modellstmEC.save("modelCNNGRUMEO.h5")


#%%############################################3GRUEC#######################################333
#######################################1DCNN+GRU+EO###################################
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Activation, Dropout, GRU
import pandas as pd
from keras.optimizers import SGD
import math
#%% LSTM+CNN for eye open dataset
temporal_dimension = X_trainEC.shape[1]
num_channels = X_trainEC.shape[2]
num_classes = 2
modellstmEC = Sequential()
modellstmEC.add(Conv1D(64, 5, activation='elu', input_shape=(temporal_dimension,num_channels)))
modellstmEC.add(MaxPool1D(3,1,))
modellstmEC.add(Dropout(0.2))
    # model.add(BatchNormalization())
modellstmEC.add(Conv1D(128, 5, activation='elu'))
modellstmEC.add(MaxPool1D(3,1,))
modellstmEC.add(Dropout(0.2))
#    # model.add(BatchNormalization())
#modellstmEC.add(Conv1D(24, 3, activation='elu'))
#modellstmEC.add(MaxPool1D(3,1,))
#modellstmEC.add(Dropout(0.2))
    # model.add(BatchNormalization())
modellstmEC.add(GRU(units=50, return_sequences=True,activation='tanh'))
modellstmEC.add(GRU(units=128, activation='tanh'))
modellstmEC.add(Dropout(0.2))
modellstmEC.add(Dense(512, activation='relu'))
#modellstmEC.add(LSTM(128)) #stacked recurrent layers said to enable deeper time series learningmodel.add(Dense(bdfData.info['nchan']))
#model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
modellstmEC.add(Dense(units=2))
modellstmEC.add(Activation('sigmoid'))
modellstmEC.summary()

# Compiling the RNN
#modelGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='binary_crossentropy',metrics=['accuracy'])
# Fitting to the training set
#history=modelGRU.fit(X_train,y_train,epochs=50,batch_size=150,validation_data=(X_test, tesy), verbose=2, shuffle=False)
#callbacks_list = [
#    keras.callbacks.ModelCheckpoint(
#        filepath='best_modellstmmulti.{epoch:02d}-{val_loss:.2f}.h5',
#        monitor='val_loss', save_best_only=True),
#    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
#]

modellstmEC.compile(loss='binary_crossentropy',
                optimizer=adam, metrics=['accuracy'])

BATCH_SIZE = 100
EPOCHS = 50

history = modellstmEC.fit(X_trainEC,
                      y_train_hotEC,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_split=0.2,
                      verbose=1)

#save.modellstmEC('modellstmEC.h5')

#import matplotlib.pyplot as plt
#plt.figure(figsize=(6, 4))
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, color='red', label='Training loss')
#plt.plot(epochs, val_loss, color='green', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs',fontsize=10)
#plt.ylabel('Loss',fontsize=10)
#plt.legend()
#plt.savefig("Losseseclstmc")
#plt.show()
#
##plotting training and validation accuracy
#plt.figure(figsize=(6, 4))
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#plt.plot(epochs, acc, color='red', label='Training acc')
#plt.plot(epochs, val_acc, color='green', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs',fontsize=10)
#plt.ylabel('Accuracy',fontsize=10)
#plt.legend()
#plt.savefig("accuracyeclstmc")
#plt.show()
#%accuracy on test data
#y_testEC = np_utils.to_categorical(X_trainEO,2)
score = modellstmEC.evaluate(X_trainEC, y_train_hotEC, verbose=1)

print('\nAccuracy on train data: %0.2f' % score[1])
print('\nLoss on train data: %0.2f' % score[0])

#y_testEC = np_utils.to_categorical(X_trainEO,2)
#y_test_hotEO=np_utils.to_categorical(y_testEO, 2)
score = modellstmEC.evaluate(X_testEC, y_test_hotEC, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])
#Accuracy on test data: 0.95
#%##################        Print confusion matrix for training data and testing datasets
#LABELS = ['Control','MDD']
#def show_confusion_matrix(validations, predictions):
#
#    matrix = metrics.confusion_matrix(validations, predictions)
#    plt.figure(figsize=(2, 2))
#    sns.heatmap(matrix,
#                cmap='coolwarm',
#                linecolor='white',
#                linewidths=1,
#                xticklabels=LABELS,
#                yticklabels=LABELS,
#                annot=True,
#                fmt='d')
#    plt.title('Confusion Matrix')
#    plt.ylabel('True Label')
#    plt.xlabel('Predicted Label')
#    plt.show()


y_pred_train = modellstmEC.predict(X_trainEC)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
y_train_hotEO1=np.argmax(y_train_hotEC, axis=1)
print(classification_report(y_train_hotEO1, max_y_pred_train))

y_pred_test = modellstmEC.predict(X_testEC)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test_hotEC, axis=1)
print(classification_report(max_y_test, max_y_pred_test))
#show_confusion_matrix(max_y_test, max_y_pred_test)
modellstmEC.save("modelCNNGRUMECnorm.h5")

from keras import backend as K
for l in range(len(modellstmEC.layers)):
    print(l, modellstmEC.layers[l])

getFeature = K.function([modellstmEC.layers[0].input, K.learning_phase()],
                        [modellstmEC.layers[9].output])

X = getFeature([X_trainEC, 0])[0]
print(X.shape)
Y=y_train_hotEC
print(Y.shape)

from sklearn.model_selection import train_test_split
X_traindd, X_testdd, y_traindd, y_testdd = train_test_split(X, Y, test_size=0.2, stratify=Y)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

parameters = {"n_neighbors": [1, 5, 10, 30],
              "weights": ['uniform', 'distance'],
              "metric": ['minkowski','euclidean','manhattan'],
              "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}
kclf = KNeighborsClassifier()
kgclf = GridSearchCV(kclf, param_grid=parameters)

kgclf.fit(X_traindd, y_traindd)

kclf = kgclf.best_estimator_
kclf.fit(X_traindd, y_traindd)

y_testKNN = kclf.predict(X_testdd)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#print_cmx(y_testdd.T[0], y_testKNN)
print(classification_report(y_testdd, y_testKNN))
print("Accuracyknn: {0}".format(accuracy_score(y_testdd, y_testKNN)))

#############################SVM###########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_traindd, y_traindd).decision_function(X_testdd)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_score_predic=classifier.predict(X_testdd)
print(classification_report(y_testdd, y_score_predic))
print("Accuracysvm: {0}".format(accuracy_score(y_testdd, y_score_predic)))

#########################################RF################################
#
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
#
#parameters = {"max_depth": [3, None],
#              "max_features": [1, 3, 10],
#              "min_samples_split": [1.0, 3, 10],
#              "min_samples_leaf": [1, 3, 10],
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"],
#              "n_estimators": [10, 20, 50]}
#rclf = RandomForestClassifier()
#rgclf = GridSearchCV(rclf, param_grid=parameters)
#
#rgclf.fit(X_traindd, y_traindd)
#
#rclf = rgclf.best_estimator_
#rclf.fit(X_traindd, y_traindd)
#
#y_testRF = rclf.predict(X_testdd)
#
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#
##print_cmx(test_label.T[0], y_testRF)
#print(classification_report(y_testdd, y_testRF))
#print("Accuracy: {0}".format(accuracy_score(y_testdd, y_testRF)))



