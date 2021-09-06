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
import copy

from hybridNet import ConvNet


import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_classes = 10
learning_rate = 0.0003
weight_decay=0.003
batch_size = 200
dropout_prob = 0.0;
#load the data
targets = torch.load("data/shuffTargTens.pt")
data = torch.load("data/shuffDataTens.pt")


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
AE_loss_fn = nn.MSELoss()
net = ConvNet(num_classes=num_classes, dropout=dropout_prob).to(device)
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
            X_epochN = addNoise(X_epoch, noise)
            X_epochN = setNorm(X_epochN)
        else:
            X_epochN = X_epoch 
            
        running_loss = 0.0
        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_epochN[s:e].unsqueeze(1).type(torch.FloatTensor)
            inputsUN = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
            inputsUN = Variable(inputsUN.cuda(0))
            optimizer.zero_grad()        
            AEout, CLout = net(inputs)       
            CLloss= loss_fn(CLout, labels)
            AEloss = AE_loss_fn(AEout, inputsUN)
            
            loss = 0.4 * AEloss + 0.6 * CLloss
            loss.backward()
            optimizer.step()
            
            
            
            running_loss += float(loss.item())
            del loss
            del labels
            del inputs 
            del AEout
            del CLout
            del CLloss
            del AEloss
            del inputsUN
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
x_in, y_in, shuffPerm = shuffleData(data[55000:], targets[55000:])


def walkBackTrain(net, optimizer, num_epochs, noise, walkbacks):
    trainAcc = []
    altAcc = []
    altNetAcc = []
    testAcc = []
    altNets = []
    X_train = data[:55000]   # perform the text/train split 
    Y_train = targets[:55000]
    
    X_alt = x_in[:5000]
    Y_alt = y_in[:5000]
    X_test = x_in[5000:]
    Y_test = y_in[5000:]

    for epoch in range(walkbacks):
        print("\n Epoch: ", epoch)
        
        X_epoch, Y_epoch, shufPerm = shuffleData(X_train, Y_train) 
        
        if noise != 0:
            X_epochN = addNoise(X_epoch, noise)
            X_epochN = setNorm(X_epochN)
        else:
            X_epochN = X_epoch 
            
        running_loss = 0.0
        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_epochN[s:e].unsqueeze(1).type(torch.FloatTensor)
            inputsUN = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
            inputsUN = Variable(inputsUN.cuda(0))
            optimizer.zero_grad()        
            AEout, CLout = net(inputs)       
            CLloss= loss_fn(CLout, labels)
            AEloss = AE_loss_fn(AEout, inputsUN)
            
            loss = 0.45 * AEloss + 0.55 * CLloss
            loss.backward()
            optimizer.step()
            
            
            
            running_loss += float(loss.item())
            del loss
            del labels
            del inputs 
            del AEout
            del CLout
            del CLloss
            del AEloss
            del inputsUN
        params = ["acc", "auc", "fmeasure"]
        print(params)
        print("Training Loss ", running_loss)
        print("Training Loss ", running_loss)
        print("train:")
        trainAcc.append(testModel(net, X_train, Y_train))
        
        print("currMod testSet:")
        altAcc.append(testModel(net, X_alt, Y_alt))
        testAcc.append(testModel(net, X_test, Y_test))
        altNets.append(copy.deepcopy(net))
        altNetAcc.append(testModel(altNets[epoch], X_alt, Y_alt))
    
    reRun = False
    reRunEp = 0
    past_acc = 0.0
    max_acc = 0.0
    print("net track size:", len(altNets))
    for epoch in range(walkbacks, walkbacks + num_epochs):
        print("\n Epoch: ", epoch)
        
        X_epoch, Y_epoch, shufPerm = shuffleData(X_train, Y_train) 
        print("first label: ", Y_epoch[0])
        if noise != 0:
            X_epochN = addNoise(X_epoch, noise)
            X_epochN = setNorm(X_epochN)
        else:
            X_epochN = X_epoch 
            
        running_loss = 0.0
        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_epochN[s:e].unsqueeze(1).type(torch.FloatTensor)
            inputsUN = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
            inputsUN = Variable(inputsUN.cuda(0))
            optimizer.zero_grad()        
            AEout, CLout = net(inputs)       
            CLloss= loss_fn(CLout, labels)
            AEloss = AE_loss_fn(AEout, inputsUN)
            
            loss = 0.4 * AEloss + 0.6 * CLloss
            loss.backward()
            optimizer.step()
            
            
            
            running_loss += float(loss.item())
            del loss
            del labels
            del inputs 
            del AEout
            del CLout
            del CLloss
            del AEloss
            del inputsUN
        params = ["acc", "auc", "fmeasure"]
        print(params)
        print("Training Loss ", running_loss)
        print("train:")
        trainAcc.append(testModel(net, X_train, Y_train))
        print("currMod altset:")
        altAcc.append(testModel(net, X_alt, Y_alt))
        print("oldMod altset:")
        altNetAcc.append(testModel(altNets[0], X_alt, Y_alt))
        print("currMod testSet:")
        testAcc.append(testModel(net, X_test, Y_test))
        
        meanAcc = 0.0
        curr_min = 100
        for i in range(walkbacks):
            meanAcc += altAcc[-1 * (i + 2)]
            if altNetAcc[-1 * (i + 2)] < curr_min:
                curr_min = altAcc[-1 * (i + 2)]
        
        pre_mean = meanAcc / walkbacks
        #pre_mean = pre_mean - 0.1 * (pre_mean - curr_min)
        print("running mean acc adj: ", pre_mean)
        
        if pre_mean > max_acc:
            max_acc = pre_mean
            
        if not reRun and pre_mean < max_acc:
            print("walking back")
            net = copy.deepcopy(altNets[0])
            optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)
            reRun = True
            reRunEp = epoch
            past_acc = pre_mean 
        else:
            print("walking forward, netsTracked: ", len(altNets))
            if reRun and altAcc[-1] < past_acc:
                print("bad start")
                net = copy.deepcopy(altNets[0])
                optimizer = Adam(net.parameters(), lr=learning_rate * 3, weight_decay = weight_decay) #over the hedge
                epoch -= 1
            else:
                optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)
                altNets.pop(0)
                altNets.append(copy.deepcopy(net))
                if reRun and epoch >= reRunEp + walkbacks:
                    reRun = False
                
        
    return trainAcc, testAcc

def treeTrain(net, optimizer, num_epochs, noise, walkbacks, depth, branches):
    print("wah")

def testModel(net, X_test, Y_test):
    with torch.no_grad():
        test_loss = 0.0
        test_total = 0
        for i in range(int(len(X_test)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_test[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_test[s:e]
            AEout, outputs = net(inputs.cuda(0))       
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device='cpu')
            total = len(labels)
            correct = (predicted == labels).sum().item()
            test_loss += correct
            test_total += total
            del inputs
            del labels
            del _
            del AEout 
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
