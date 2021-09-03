
""" 
This program was built to compare the accuracies on the dataset after excluding different pairs of symmertrical channels, one at a time. 7 different datasets were 
created using newDataloader.py. 
For each of these datasets, the program prints out the accuracies on the train and test sets. 
The mainLoop() function is the main function that specifies the hyperparameters used during training. The accuracies reported below are for the following set of 
parameters: learning rate = 0.0003, weight decay=0.003, batch size=50, number of epochs = 30, no noise. 
"""

from newNet import newNet
import torch 
import torch.nn as nn   
from torch.autograd import Variable
import numpy as np
from torch.optim import Adam    #importing an iptimizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from convNet import ConvNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

loss_list = []
iteration_list = []
accuracy_list = []
count = 0
min_val_loss = np.Inf
val_array = []
correct = 0
iter = 0
count = 0
iter_array = []
loss_array = []
total = 0
accuracy_array = []



# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True



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
    randomAdd = np.random.normal(0, stdDev, dataset.shape)
    noisedData = torch.tensor(dataset.numpy() + randomAdd).type(torch.FloatTensor)
    return noisedData


def createTrainTestSplit(data):
    data = setNorm(data)   # normalize the data
    X_train = data[:60000]   
    Y_train = targets[:60000]
    X_test = data[60000:]
    Y_test = targets[60000:]
    return X_train, Y_train, X_test, Y_test


"""
The main training function for our model. Randomizes dataset on each epoch and adds noise if noise param !=0. 
    net: the model to train and test
    optimizer: specified optimizer
    num_epoch: number of epoch to run during training 
    noise: proportion of Gaussian noise with input standard deviation to dataset
    returns lists of accuracy at each epoch on training and test sets.
"""
def trainModel(net, data, optimizer, loss_fn, num_epochs, noise, batch_size, implement_early_stop = False, printout = False):
    trainAcc, testAcc = [], []
    n_epochs_stop = 6
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.Inf
    count, correct, iter, count, total = 0, 0, 0, 0, 0
    teration_list, accuracy_list, loss_list, val_array, iter_array, loss_array, accuracy_array = [], [], [], [], [], [],[]
    
    writer = SummaryWriter()
    X_train, Y_train, X_test, Y_test = createTrainTestSplit(data)
    for epoch in range(num_epochs):
        if printout:
            print("\n Epoch: ", epoch)
        X_epoch, Y_epoch, shufPerm = shuffleData(X_train, Y_train) 
        
        if noise != 0:
            X_epoch = addNoise(X_epoch, noise)
            X_epoch = setNorm(X_epoch)
        running_loss = 0.
        
        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
    
            inputs = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
        
            optimizer.zero_grad() # clear gradients       
            outputs = net(inputs)  # forward propagation
            loss= loss_fn(outputs, labels)   # calculate the loss
            loss.backward()   # Calculating gradients
            optimizer.step()  # Update parameters

            running_loss += float(loss.item())

        del loss 
        del labels
        del inputs 
        del outputs 
        curr_trainAcc = testModel(net, X_train, Y_train, batch_size)
        curr_testAcc = testModel(net, X_test, Y_test, batch_size)

        if printout:
            print("Training Loss:", running_loss)
            print("Accuracy on the train set: {} %".format(curr_testAcc))
            print("Accuracy on the test set: {}%".format(curr_trainAcc))
        
        trainAcc.append(curr_trainAcc)
        testAcc.append(curr_testAcc)

        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("trainAcc", curr_trainAcc, epoch)
        writer.add_scalar("testAcc", curr_testAcc, epoch)

    return trainAcc, testAcc
    


""" 
Model testing function, prints accuracy of classifier on input test data and labels
    net: trained model to test
    X_test, Y_test: data and labels, respectively
"""
def testModel(net, X_test, Y_test, batch_size):
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
    return (100 * test_loss / test_total)

        
""" 
Calculates precision, recall and f-scores for the model
    net: trained model to test
    X_test, Y_test: data and labels, respectively
    returns  precision, recall and f-scores for the model
"""        
def genScores(net, X_test, Y_test):
    preds, _ = generateConfusionMatrix(net, X_test, Y_test)
    return precision_recall_fscore_support(Y_test.numpy(), preds, average='micro')



def mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 30, noise = 0.0):
    writer = SummaryWriter()
    # Device configuration

    # Hyper parameters
    num_classes = 10
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    net = newNet(num_classes=num_classes,dropout=0.0).cuda(0)
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)

    trAcc, teAcc = trainModel(net, data, optimizer, loss_fn, num_epochs, noise, batch_size, implement_early_stop=True)
    writer.flush()
    return trAcc, teAcc




data = torch.load('/scratch/akazako1/10DigData_noO1O2.pt')   # location of the dataset 
targets = torch.load('/scratch/akazako1/10DigTarg_noO1O2.pt')    
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 30, noise = 0)
print("train acc & test acc for dataset with O1, O2 removed is ", trAcc, teAcc)

# Maximum accuracy:
# Accuracy after 30 epoch


data = torch.load('/scratch/akazako1/10DigData_noP7P8.pt')
targets = torch.load('/scratch/akazako1/10DigTarg_noP7P8.pt') 
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 50, noise = 0)
print("train acc & test acc for dataset with P7, P8 removed is ", trAcc, teAcc)


# Maximum accuracy:
# Accuracy after 30 epoch:


data = torch.load('/scratch/akazako1/10DigData_noF7F8.pt')
targets = torch.load('/scratch/akazako1/10DigTarg_noF7F8.pt') 
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 50, noise = 0)
print("train acc & test acc for dataset with F7, F8 removed is ", trAcc, teAcc)

# Maximum accuracy:
# Accuracy after 30 epoch:


data = torch.load('/scratch/akazako1/10DigData_noAF3AF4.pt')
targets = torch.load('/scratch/akazako1/10DigTarg_noAF3AF4.pt') 
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 50, noise = 0)
print("train acc & test acc for dataset with AF3, AF4 removed is ", trAcc, teAcc)

# Maximum accuracy:
# Accuracy after 30 epoch:


data = torch.load('/scratch/akazako1/10DigData_noFC5FC6.pt')
targets = torch.load('/scratch/akazako1/10DigTarg_noFC5FC6.pt') 
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 50, noise = 0)
print("train acc & test acc for dataset with FC5, FC6 removed is ", trAcc, teAcc)

# Maximum accuracy:
# Accuracy after 30 epoch


data = torch.load('/scratch/akazako1/10DigData_noT7T8.pt')
targets = torch.load('/scratch/akazako1/10DigTarg_noT7T8.pt') 
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 50, noise = 0)
print("train acc & test acc for dataset with T7, T8 removed is ", trAcc, teAcc)

# Maximum accuracy:
# Accuracy after 30 epoch


data = torch.load('/scratch/akazako1/10DigData_noF3F4.pt')
targets = torch.load('/scratch/akazako1/10DigTarg_noF3F4.pt') 
trAcc, teAcc = mainLoop(data, targets, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 50, noise = 0)
print("train acc & test acc for dataset with F3, F4 removed is ", trAcc, teAcc)

# Maximum accuracy:
# Accuracy after 30 epoch:
