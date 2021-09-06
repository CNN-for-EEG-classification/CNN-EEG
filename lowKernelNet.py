import torch 
import torch.nn as nn   

class lkNet(nn.Module):  #define a new class named SimpleNet, which extends the nn.Module class
    def __init__(self, num_classes=10, dropout=0.0):
        super(lkNet, self).__init__()
        self.layer1 = nn.Sequential(
           nn.ZeroPad2d((20,20,0,0)),
           nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = (1,41), stride = (1,1), padding = 0),
           nn.LeakyReLU(),
           nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = (2,1), stride = (2,1), padding = 0),
            nn.BatchNorm2d(20, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (1,3), stride = (1,3)),
            nn.Dropout(p=dropout)
            )
        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv2d(in_channels=20, out_channels = 40, kernel_size = (1,7), stride = (1,1)),
            nn.LeakyReLU())
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d((5,5,0,0)),
            nn.Conv2d(in_channels=40, out_channels = 80, kernel_size = (1,11), stride = (1,1)),
            nn.BatchNorm2d(80, affine=True),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (7,1), stride=(7,1)),
            nn.BatchNorm2d(80, affine=True),
            nn.LeakyReLU()
            )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.linear1 = nn.Sequential(
            nn.Linear(80*4, num_classes),
            nn.LogSoftmax())
        
            
    def forward(self, x):
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