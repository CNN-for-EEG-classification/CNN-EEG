"""
Main CCN model that can achieve a mean accuracy of 17% for 10-class classification.
""" 

num_channels= 20

import torch 
import torch.nn as nn   

class ConvNet(nn.Module): 
    def __init__(self, num_classes=10, dropout=0.0):
        super(ConvNet, self).__init__()
        #encoder
        self.layer1 = nn.Sequential(
           nn.Conv2d(in_channels = 1, out_channels = num_channels, kernel_size = (1,61), stride = (1,1), padding = 0),
           nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(num_channels),
            nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = (2,1), stride = (2,1), padding = 0),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=num_channels, out_channels = num_channels*2, kernel_size = (1,31), stride = (1,1)),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(num_channels*2),
            nn.Conv2d(in_channels=num_channels*2, out_channels = num_channels*4, kernel_size = (1,21), stride = (1,1)),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels = num_channels*4, out_channels = num_channels*1, kernel_size = (7,1), stride=(7,1)),
            nn.LeakyReLU())
        
        #central Layer
        self.linear1 = nn.Sequential(
            nn.Linear(1200, 1000),
            nn.LeakyReLU())
        
        #classifier output
        self.linOut = nn.Sequential(
            nn.Linear(1000, 10),
            nn.LogSoftmax())
        
        #decoder
        self.decLayer1 = nn.Sequential(
            nn.ConvTranspose2d(1,2,(7,1)),
            nn.LeakyReLU()
            )
        self.decLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 1), stride=(1,2)),
            nn.LeakyReLU()
            )
        self.decLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1,1), stride=(1,2)),
            nn.LeakyReLU())
        self.decLayer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=(2,1), stride=(2,1)),
            nn.LeakyReLU())
        self.combLayer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,1), stride=(1,1)),
            nn.Tanh())
        
            
    def forward(self, x):
        out = self.layer1(x)
        out =self.layer2(out)
        out =self.layer3(out)
        out =self.layer4(out)
        out =self.layer5(out)
        out = torch.flatten(out, start_dim=1)
        inter = self.linear1(out)
        
        classOut = self.linOut(inter)
        
        out = self.decLayer1(inter.unsqueeze(1).type(torch.cuda.FloatTensor).unsqueeze(1).type(torch.cuda.FloatTensor))
        out = self.decLayer2(out)
        out = self.decLayer3(out)
        out = self.decLayer4(out)
        out = self.combLayer(out)
        
        return out, classOut
        
        
        
