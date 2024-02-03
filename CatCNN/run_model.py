# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2

import camera
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (input channel size, output channel size, kernel size)
        self.pool = nn.MaxPool2d(2, 2) # (kernel size, stride)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 1st param derived from ((input_width-filter_size+2*padding)/stride) + 1
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 = number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # -1 lets pytorch find right size for us
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation function at the end
        return x

def load():
    print(f"loading model on {device}")
    net = Net().to(device)
    net.load_state_dict(torch.load("model.pth", map_location=device))
    net.eval()
    print("model loaded")
    return net

        
        
if __name__ == '__main__':
    net = load()
    image = cv2.imread("meow.jpg")
    if image is not None:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((32,32)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_tensor = transform(image)
        batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        batch = batch.to(device)
        output = net(batch)
        _, output = torch.max(output,1)
        print("Prediction:", output)
        
    
        

