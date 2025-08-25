
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size= 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels= 128,kernel_size = 3 )
        self.pool2 =  nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3)


        self.fc1 = nn.Linear(in_features=51200,out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features = 64)
        self.fc3 = nn.Linear(in_features=64, out_features=7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x =  F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = torch.flatten(x,1) #flatten all dimensions except the first one 32 x (channels,length,width,)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4)

        x= self.fc3(x)

        return x
    

cnnModel = CNNModel()
