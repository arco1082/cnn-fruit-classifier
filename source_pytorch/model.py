# torch imports
import torch.nn.functional as F
import torch.nn as nn
import torch

class FruitClassifier(nn.Module):

    def __init__(self, class_count):
    
        super(FruitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1 )       
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1 )
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1 )
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1 )
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1 )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, class_count)

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_bn2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_bn3(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv_bn4(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv_bn5(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv_bn6(x)
        
        # flatten image input
        x = x.view(-1, 256 * 3 * 3)        
       
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x