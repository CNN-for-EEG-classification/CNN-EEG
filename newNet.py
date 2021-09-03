"""
An attempt to re-build the network in Keras
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms 

from skorch import NeuralNetClassifier


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score



class newNet(nn.Module): 
    def __init__(self, num_classes=10, dropout=0.0):
        super(newNet, self).__init__()
        self.layer1 = nn.Sequential(
           #nn.ZeroPad2d((15,15,0,0)),
           nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (1,9), stride = (1,1), padding = 0),
           nn.LeakyReLU(),
           nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = (2,1), stride = (2,1), padding = 0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (1,3), stride = (1,2))
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels = 80, kernel_size = (1,18), stride = (1,1)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        self.layer4 = nn.Sequential(
            #nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels = 160, kernel_size = (1,11), stride = (1,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = 160, kernel_size = (7-1,1), stride=(7-1,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU()
            )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.linear1 = nn.Sequential(
            nn.Linear((160)*4, num_classes),
            #nn.Softmax(dim=0)
            nn.LogSoftmax(dim=0)
            )

        
            
    def forward(self, x):
        #print("channels x height x width")
        #print("input: ", x[0].shape)
        out = self.layer1(x)
        #print("layer1: ",out[0].shape)
        out = self.layer2(out)
        #print("layer2: ",out[0].shape)
        out= self.layer3(out)
        #print("layer3: ",out[0].shape)
        out = self.pool2(out)
        #print("pool2: ",out[0].shape)
        out = self.layer4(out)
        #print("layer4: ",out[0].shape)
        out = self.pool3(out)
        #print("pool3: ",out[0].shape)
        out = self.layer5(out)
        #print("layer5: ",out[0].shape)
        out = self.pool4(out)
        #print("pool4: ",out[0].shape)
        out = torch.flatten(out,start_dim=1)
        #print("flattened: ",out[0].shape)
        out= self.linear1(out)
        return out

"""

# Hyper parameters
num_classes = 10
learning_rate = 0.0003
weight_decay=0.003
batch_size = 50



#load the data
targets = torch.load("/scratch/akazako1/10DigTarg.pt")
data = torch.load("/scratch/akazako1/10DigData.pt")


DEVICE = torch.device("cpu")

y_train = np.array([y for x, y in iter(train_dataset)])  #

torch.manual_seed(0)

net = NeuralNetClassifier(
    ConvNet,
    max_epochs=10,
    iterator_train__num_workers=4,
    iterator_valid__num_workers=4,
    lr=1e-3,
    batch_size=64,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=DEVICE
)



net.fit(X_train.unsqueeze(1).type(torch.FloatTensor), Y_train)


val_loss=[]
train_loss=[]
for i in range(10):
    val_loss.append(net.history[i]['valid_loss'])
    train_loss.append(net.history[i]['train_loss'])
    
plt.figure(figsize=(10,8))
plt.semilogy(train_loss, label='Train loss')
plt.semilogy(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig("train_val_loss_batch64_lr0003_epoch10" + str(batch_size)+ ".png")
plt.show()
plt.close("all")


y_pred = net.predict(test_dataset)
accuracy_score(y_test, y_pred)
 """