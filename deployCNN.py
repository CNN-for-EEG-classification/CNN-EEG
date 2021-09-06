#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file to run all of our operations for multi class classification.
"""

import torch 
import torch.nn as nn   
from torch.autograd import Variable
import numpy as np
from torch.optim import Adam    #importing an iptimizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from convNet import ConvNet
from thinNet import thinNet
from lowKernelNet import lkNet


import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_classes = 10
learning_rate = 0.0003
weight_decay=0.003
batch_size = 50

#load the data
targets = torch.load("shuffTargTens.pt")
data = torch.load("shuffDataTens.pt")


#shuffles data, used before each epoch in train model
def shuffleData(inData, labels):
    reshuffleIndex = np.random.permutation(len(inData))
    return torch.tensor(inData.numpy()[reshuffleIndex]), torch.tensor(labels.numpy()[reshuffleIndex]), reshuffleIndex


#normalizes dataset so every EEG channel has a mean of 0 and standard deviation of 1 across all examples.
def setNorm(dataset):
    for i in range(dataset.shape[1]):
        channelMeans = dataset[:,i,:].mean()
        channelStdDev = dataset[:,i,:].std()
        dataset[:,i,:] = (dataset[:,i,:] - channelMeans)/channelStdDev
    return dataset

#Adds Gaussian noise with input standard deviation to dataset.
def addNoise(dataset, stdDev):
    print("Ayy lmao")
    randomAdd = np.random.normal(0, stdDev, dataset.shape)
    noisedData = torch.tensor(dataset.numpy() + randomAdd).type(torch.FloatTensor)
    return noisedData

# Define the loss function and optimizer
loss_fn = nn.NLLLoss()
net = ConvNet(num_classes=num_classes,dropout=0.0).cuda(0)
optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)


data = setNorm(data)   #normalize the data
X_train = data[:60000]   # perform the text/train split 
Y_train = targets[:60000]
X_test = data[60000:]
Y_test = targets[60000:]

"""
Training function for our model, randomizes dataset on each epoch and adds noise if noise param !=0. 
    net: the model to train and test
    optimizer: specified optimizer
    num_epoch: number of epoch to run during training 
    noise: proportion of Gaussian noise with input standard deviation to dataset
    returns lists of accuracy at each epoch on training and test sets.
"""
def trainModel(net, optimizer, num_epochs, noise):
    trainAcc = []
    testAcc = []
    for epoch in range(num_epochs):
        print("\n Epoch: ", epoch)
        
        X_epoch, Y_epoch, shufPerm = shuffleData(X_train, Y_train) 
        
        if noise != 0:
            X_epoch = addNoise(X_epoch, noise)
            X_epoch = setNorm(X_epoch)
        running_loss = 0.0
        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
        
            optimizer.zero_grad()        
            outputs = net(inputs)       
            loss= loss_fn(outputs, labels)
            loss.backward()
    
            optimizer.step()
    
            running_loss += float(loss.item())
            del loss
            del labels
            del inputs 
            del outputs 
        params = ["acc", "auc", "fmeasure"]
        print(params)
        print("Training Loss ", running_loss)
        trainAcc.append(testModel(net, X_train, Y_train))
        testAcc.append(testModel(net, X_test, Y_test))
    return trainAcc, testAcc
    


""" 
Model testing function, prints accuracy of classifier on input test data and labels
    net: trained model to test
    X_test, Y_test: data and labels, respectively
"""
def testModel(net, X_test, Y_test):
    with torch.no_grad():
        test_loss = 0.0
        test_total = 0
        for i in range(int(len(X_test)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_test[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_test[s:e]
            outputs = net(inputs.cuda(0))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device='cpu')
            total = len(labels)
            correct = (predicted == labels).sum().item()
            test_loss += correct
            test_total += total
            del inputs
            del labels
            del _
    print('Test Accuracy of the model on the test set is: {} %'.format(100 * test_loss / test_total))
    return (100 * test_loss / test_total)

"""
Generates a confusion matrix for an input net with the input data and labels
    net: trained model to test
    X_test, Y_test: data and labels, respectively
    returns a list of predictions, as well the the confusion matrix
"""
def generateConfusionMatrix(net, X_test, Y_test):
    preds = np.array([], dtype=int)
    predLen = 0
    with torch.no_grad():
        for i in range(int(len(X_test)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            predLen += batch_size
            inputs = X_test[s:e].unsqueeze(1).type(torch.FloatTensor)
            outputs = net(inputs.cuda(0))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device='cpu').numpy()
            preds = np.concatenate((preds, predicted))
    return preds, confusion_matrix(Y_test.numpy()[:predLen], preds, normalize='true')

"""
Produces validation curves for each dropout probability in params; 
trains a model with that value, and prints the resulting validation curve 
    numEpochs: number of epoch to run
    params: other parameters (learning_rate, weight_decay) to examine
"""
def genDropoutValCurves(numEpochs, params):
    print("Dropout testing")
    for i in range(len(params)):
        net = ConvNet(num_classes=num_classes,dropout=params[i]).cuda(0)
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)
        trAcc, teAcc = trainModel(net, optimizer, numEpochs, 0.0)
        plt.plot(range(numEpochs), trAcc, label='Train set accuracy')
        plt.plot(range(numEpochs), teAcc, label='Test set accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim(0,numEpochs)
        plt.ylim(0,100)
        plt.legend()
        plt.show()

""" 
Produces validation curves for each noise value in params; trains a model with 
that value,and prints the resulting validation curve 
    numEpochs: number of epoch to run
    params: other parameters (learning_rate, weight_decay) to examine
""" 
def genNoiseValCurves(numEpochs, params):
    for i in range(len(params)):
        net = ConvNet(num_classes=num_classes,dropout=0.0).cuda(0)
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)
        trAcc, teAcc = trainModel(net, optimizer, numEpochs, params[i])
        plt.plot(range(numEpochs), trAcc, label='Train set accuracy')
        plt.plot(range(numEpochs), teAcc, label='Test set accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim(0,numEpochs)
        plt.ylim(0,100)
        plt.legend()
        plt.show()

        
""" 
Calculates precision, recall and f-scores for the model
    net: trained model to test
    X_test, Y_test: data and labels, respectively
    returns  precision, recall and f-scores for the model
"""        
def genScores(net, X_test, Y_test):
    preds, _ = generateConfusionMatrix(net, X_test, Y_test)
    return precision_recall_fscore_support(Y_test.numpy(), preds, average='micro')


def genTSplots(ts, title):
    plt.rc('font', size=12)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Specify how our lines should look
    ax.plot(range(len(ts)), ts, color='tab:orange')

    # Same as above
    ax.set_xlabel('time')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title(title)
    ax.grid(True)
    plt.show()

# Save the model checkpoint
#torch.save(net.state_dict(), 'model.ckpt')
        
#genNoiseValCurves(25, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
#genDropoutValCurves(25, [0.0,0.1,0.2,0.3,0.4,0.5])
