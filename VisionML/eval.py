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


    _, test = random_split(dataset, [0.8,0.2])

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
        num_correct = 0
        num_samples = 0

        for i, (inputs, labels) in enumerate(valloader,0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            value, predicted = torch.max(outputs, 1)

            for i in range(batch_size):
                label = labels[i]
                prediction = predicted[i]
                print(f"{prediction}\t{label}")

            # max_value, max_index = torch.max(outputs.view(-1), dim=0)

            # # Get the entire row with the highest confidence value
            # row_with_highest_confidence = outputs[max_index // outputs.shape[1], :]

            # print(row_with_highest_confidence.tolist(), labels)

            # print(outputs)


            num_samples += labels.size(0)
            num_correct += (outputs == labels).sum().item()

        acc = 100.0 * num_correct / num_samples
        print(f'Accuracy of the network: {acc}%')


if __name__ == '__main__':
    main()