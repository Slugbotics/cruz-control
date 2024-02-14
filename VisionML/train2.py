import os
import sys
import torch
import torch.nn as nn
from model2 import LaneCNN
import torch.optim as optim
from dk_loader import DataCC
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

learning_rate = 0.001
batch_size = 64
epochs = 10

def train():
    path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])

    dataset = DataCC("dc_dataset_2", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    # multithreaded data loading
    trainloader = DataLoader(train, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=batch_size, 
                       shuffle=False, num_workers=2)
    
    net = LaneCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print(f"Training on {device}")

    for epoch in range(epochs):
        net.train()
        for inputs, (steering_targets, throttle_targets) in trainloader:
            inputs = inputs.to(device)
            steering_targets = steering_targets.to(device)
            throttle_targets = throttle_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            steering_preds, throttle_preds = net(inputs)
            
            # Compute loss
            steering_loss = criterion(steering_preds, steering_targets)
            throttle_loss = criterion(throttle_preds, throttle_targets)
            total_loss = steering_loss + throttle_loss

            # Backward pass
            total_loss.backward()

            # Update weights
            optimizer.step()

        # Validation
        net.eval()
        with torch.no_grad():
            for val_inputs, (val_steering_targets, val_throttle_targets) in valloader:
                val_inputs = val_inputs.to(device)  # Move validation data to GPU
                val_steering_targets = val_steering_targets.to(device)
                val_throttle_targets = val_throttle_targets.to(device)

                val_steering_preds, val_throttle_preds = net(val_inputs)
                val_steering_loss = criterion(val_steering_preds, val_steering_targets)
                val_throttle_loss = criterion(val_throttle_preds, val_throttle_targets)
                val_total_loss = val_steering_loss + val_throttle_loss

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}, Validation Loss: {val_total_loss.item():.4f}')
        torch.save(net.state_dict(), os.path.join("models", f"{path}_e{epoch+1}.pth"))


    print("Finished training") 
    torch.save(net.state_dict(), os.path.join("models", f"{path}_final.pth"))

if __name__ == '__main__':
    train()
            