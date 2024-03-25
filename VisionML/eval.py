import os
import sys
import torch
import torchvision
import torch.nn as nn
from model2 import LaneCNN
from dk_loader import DataCC
from torch.utils.data import random_split, DataLoader
from camera import VideoCapture
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

def main():

    model_path = os.path.join("models",f"{sys.argv[1]}.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eval on {device}")
    img_transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])

    model = LaneCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cam = VideoCapture(0)

    model.eval()

    while True:

        frame = cam.read()

        image = Image.fromarray(frame)
        input_image = img_transform(image).unsqueeze(0).to(device)

        prediction = model(input_image)

        print(prediction)


if __name__ == '__main__':
    main()