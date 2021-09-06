import torch 
import torch.nn as nn   

class thinNet(nn.Module):  #define a new class named SimpleNet, which extends the nn.Module class
    def __init__(self, num_classes=10, dropout=0.0):
        super(thinNet, self).__init__()
        self.layer1 = nn.Sequential(
           nn.ZeroPad2d((5,5,0,0)),
           nn.Conv2d(in_channels = 1, out_channels = 12, kernel_size = (1,11), stride = (1,1), padding = 0),
           nn.LeakyReLU(),
           nn.Dropout(p=dropout),
           nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels = 24, kernel_size = (2,11), stride = (2,1)),
            nn.BatchNorm2d(24, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.layer4 = nn.Sequential(
            nn.ZeroPad2d((5,5,0,0)),
            nn.Conv2d(in_channels=24, out_channels = 48, kernel_size = (1,11), stride = (1,1)),
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (4,1), stride=(1,1)),
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU()
            )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2)))
        self.linear1 = nn.Sequential(
            nn.Linear(48*9, num_classes),
            nn.LogSoftmax())
        
            
    def forward(self, x):
        out = self.layer1(x)
        out= self.layer3(out)
        out = self.pool2(out)
        out = self.layer4(out)
        out = self.pool3(out)
        out = self.layer5(out)
        out = self.pool4(out)
        out = torch.flatten(out,start_dim=1)
        out= self.linear1(out)
        return out
