import os
import sys
import torch
import torchvision
import torch.nn as nn
from model2 import LaneCNN
from dk_loader import DataCC
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms

def main():
    steering_error = 0
    throttle_error = 0
    batch_size = 1

    model_path = os.path.join("models",f"{sys.argv[1]}.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eval on {device}")
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])



    dataset = DataCC("dc_dataset_1", transform=transform)
    train_size = int(0.3 * len(dataset))
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    # multithreaded data loading
    valloader = DataLoader(val, batch_size=batch_size, 
                        shuffle=False, num_workers=2)

    model = LaneCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    with torch.no_grad():

        for j, (inputs, (steering_targets, throttle_targets)) in enumerate(valloader,1):
            inputs = inputs.to(device)
            steering_targets = steering_targets.to(device)
            throttle_targets = throttle_targets.to(device)

            steering_preds, throttle_preds = model(inputs)

            
            for i in range(batch_size):
                targets = [steering_targets[i], throttle_targets[i]]
                prediction = [steering_preds[i], throttle_preds[i]]
                steering_error += prediction[0].item() - targets[0].item()
                throttle_error += prediction[1].item() - targets[1].item()

                print(f"Predicted: [{prediction[0].item():6.3f},{prediction[1].item():6.3f}]   Target: [{targets[0].item():6.3f},{targets[1].item():6.3f}]   Steering Error: {prediction[0].item() - targets[0].item():6.3f}   Throttle Error: {prediction[1].item() - targets[1].item():6.3f}   Average steering error: {(steering_error/j)*100:6.3f}%   Average throttle error: {(throttle_error/j)*100:6.3f}%")


if __name__ == '__main__':
    main()