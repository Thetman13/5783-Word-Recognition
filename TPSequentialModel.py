#Tyler Porter
#Neural Network using Keras to learn character (letter) recognition.


#10/17 19:46
#Currently having issues fitting test data to output. I think we'll need to
#convert the list of characters in the data set to a 2d array where each iteration
#is a new letter, and each letter is represented by only 1 '1' appearing in the
#output layer with 26 nodes. (25 being 0)

#10/18 20:11
#When using testing data rather than training data the set sdo not currently match
#I will need to rework how those sets are made in the load function

#imports
import numpy as np
import keras as ks
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
import pandas as pd


#global variables
  #intaking letter data
data = []
  #labels is the letter data for the given iteration
labels = []
  #pixels is the 7x15 representation of pixels making up each letter.
  #each iteration is a new letter
pixels = [[], []]
pixelstesting = [[], []]
trainingvalues = np.zeros((40000, 26))
testingvalues = np.zeros((13000, 26)) #less than 13000 needed





#loads data into global variables from csv
def Load():
    global data, labels, pixels
    data = pd.read_csv("letter.data", sep = '\t')
    labels = data.iloc[:, 1]
    #convert labels to 52k 26 long binary vectors, of which only 1 value is
    #activated representing letter selection
    for i in range(len(labels)):
        temp = ord(labels[i])
        temp = temp - 97 #offset from ASCII values
        if (i < 40000):
            trainingvalues[i][temp] = 1
        else:
            testingvalues[i-40000][temp] = 1

    pixels = data.iloc[0:40000, 6:111]
    pixelstesting = data.iloc[40001:, 6:111]



def BuildModel():
    #This will build every layer of the model

    acfun = "sigmoid"
    #This will be a sequential model.
    #Each layer will have 2/3rds the nodes until reaching 26 for output
    model = Sequential()
    #First an input layer taking the 15x7 grid of binary values
    model.add(Dense(70, input_dim = 105, activation=acfun))
    model.add(Dense(48, activation=acfun))
    model.add(Dense(32, activation=acfun))
    model.add(Dense(26, activation=acfun))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

#Will fit the model to training data and train model
#Needs to be much more complex
def FitModel(model):
    global pixels, trainingvalues
    model.fit(pixels, trainingvalues, epochs = 40, batch_size=100, verbose=1)
    return model

def TestModel(model):
    global testingvalues, pixelstesting
    testacc = model.evaluate(pixelstesting, testingvalues, verbose=0)
    print(testacc)



#main
Load()
model = BuildModel()
model = FitModel(model)
model.summary

#Test model currently does not have matching X and Y sets

#TestModel(model)
