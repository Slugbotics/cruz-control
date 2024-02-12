import os
import sys
import torch
import torch.nn as nn
from model import LaneCNN
import torch.optim as optim
from dk_loader import DataCC
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def main():
    save_path = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    learning_rate = 0.01
    batch_size = 64
    epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),  # Resize images to match the input size
    ])

    dataset = DataCC("dc_dataset_1", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])




    model = LaneCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("training started")
    for epoch in range(epochs):
        for images, labels in train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Save the final model
    torch.save(model.state_dict(), os.path.join("models", f"{save_path}_final.pth"))

if __name__ == '__main__':
    main()