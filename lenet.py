import torch
import torch.nn as nn
from torch.optim import optimizer 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.optim as optim 


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2), padding = (0,0))
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size=(5,5), padding=(0,0), stride=(1,1))
        self.linear1 = nn.Linear(in_features = 120, out_features =84)
        self.linear2 = nn.Linear(in_features = 84, out_features = 10)

    def forward(self, x):
        print('first time data shape ', x.shape)
        x = self.conv1(x)
        print('shape after conv1', x.shape)
        x = self.relu(x)
        print('shape after 1 conv and relu1 ', x.shape)
        x = self.pool(x)
        print('shape after 1 conv 1 relu and 1 pool', x.shape)
        #second convolution
        x = self.conv2(x)
        x = self.relu(x)
        print('shape after 2 conv and 2 relu', x.shape)
        x = self.pool(x)
        #third convolution layer 
        print('shape of x ', x.shape)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        #linear 
        x = self.linear1(x)
        x = self.relu(x)
        #linear2
        x = self.linear2(x)

        return x 

x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)
        











