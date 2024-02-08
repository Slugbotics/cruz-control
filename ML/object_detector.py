import torch
import torch.nn as nn
from torch.optim as optim
from torchvision import datasets, transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class laneDetector(nn.Module): # add/delete conv layers as you see fit, test model to determine
  def __init__(self, num_classes):
    super(laneDetector, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self.relu1=nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.relu2=nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.relu3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(256 * 28 * 28, 512)
    self.relu4 = nn.ReLU()
    self.fc2 = nn.Linear(512, num_classes)
    
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
        
        
  

        




