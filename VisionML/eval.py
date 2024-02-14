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
    batch_size = 64

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
                        shuffle=False, num_workers=1)

    model = LaneCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    with torch.no_grad():

        for inputs, (steering_targets, throttle_targets) in valloader:
            inputs = inputs.to(device)
            steering_targets = steering_targets.to(device)
            throttle_targets = throttle_targets.to(device)

            steering_preds, throttle_preds = model(inputs)

            
            for i in range(batch_size):
                targets = [steering_targets[i], throttle_targets[i]]
                prediction = [steering_preds[i], throttle_preds[i]]
                print(f"{prediction}\t{targets}")


if __name__ == '__main__':
    main()