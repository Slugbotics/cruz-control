# TRAINING A CNN


# TYPICAL CNN ARCHITECTURE
# NOTE: Pooling layers are used to automatically learn certain features on the input images
# NOTE: When defining the convolutional layers, its important to properly define the padding, 
# or else the filter wont be able to parse through all the available regions in the image
# NOTE: Max pooling is used to down sample images by applying a maximum filter to regions.
# For example, a 2x2 max-pool will apply a maximum filter to a 2x2 region. Reduces computational cost
# reduces overfitting by providing an abstracted version of the input data


# CONV
# RELU
# CONV
# RELU
# POOL

# CONV
# RELU
# CONV
# RELU
# POOL

# FULLY CONNECTED LAYERS

import sys
import torch
import torch.nn as nn
from torch.nn import LogSoftmax


class LaneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st param: num input channels (leave as 3) 
        # 2nd param: num output channels generally we want to keep this number small in the early layers of the model
        # around 32 to 64 (to prevent overfitting)
        # 3rd param: kernel size - usually, we want smaller kernel size (3x3) in early layers to capture fine details
        # we can move up in kernel size in deeper layers to capture global details
    

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32,64,3)
        self.relu2 = nn.ReLU()
        # a larger kernel size means a more aggressive downsampling of the input dimensions
        # a smaller kernel size perserves the more fine grained details of the input feature map
        # in the early layers, you want to capture the fine details so keep kernel size small

        # stride influences the downsampling factor
        # larger stride means a more aggressive of the input spacial dimensions
        # if you want to reduce spacial dimensions more rapidly, increase stride
        # larger strides introduce overlap between neighboring regions
        # the choice of stride impacts how much info is retained from adjacent regions
        # in early layers, smaller strides are more common as we want to capture fine details
        # in deeper layers, we will increase stride to get a general idea
        #TODO print the output size of relu2 for max pooling calculations
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output size of max pooling layer - (input size - kernel size)/stride + 1


        self.conv3 = nn.Conv2d(64, 128, 5)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256,5)
        self.relu4 = nn.ReLU()
        

        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=3)
        
        self.conv5 = nn.Conv2d(256, 512, 7)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(512, 256, 7)
        self.relu6 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=3)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6400, 256)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu8 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu9 = nn.ReLU()
        self.fc4 = nn.Linear(64,2)

        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1))))
        x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
        x = self.pool2(self.relu4(self.conv4(self.relu3(self.conv3(x)))))
        x = self.pool3(self.relu6(self.conv6(self.relu5(self.conv5(x)))))
        x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.relu7(self.fc1(x))
        x = self.relu8(self.fc2(x))
        x = self.relu9(self.fc3(x))
        x = self.fc4(x)
        x = self.logSoftmax(x)
        return x
