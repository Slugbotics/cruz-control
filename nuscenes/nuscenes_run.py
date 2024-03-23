import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from PIL import Image
import sys
import torch.nn.functional as F
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from pathlib import Path

# Usage Example:
# python3 nuscenes_run.py /path/to/nuscenes/v1.0-mini v1.0-mini

if len(sys.argv) < 2:
    raise Exception('ERROR, PROVIDE PATH TO MODEL')

if len(sys.argv) < 3:
    raise Exception('ERROR, PROVIDE version type')

PATH = sys.argv[1]
VERSION = sys.argv[2]

nusc = NuScenes(version=VERSION, dataroot=PATH, verbose=False)
nusc_can = NuScenesCanBus(dataroot=PATH)


def get_closest_can(time, can_objects):
    closest = {}
    prev_diff = 1000000 # 1 Second in microseconds
    for object in can_objects:
        diff = object["utime"] - time
        if diff > 0 and diff < prev_diff:
            closest = object
            prev_diff = diff
    # print("Time difference: ", prev_diff)
    return closest

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def normalize_vehicle_monitor_can(can_obj):
    new_obj = {}

    min_brake = 0
    max_break = 126

    min_steering = -780
    max_steering = 779.9

    min_throttle = 0
    max_throttle = 1000

    new_obj["brake"] = num_to_range(can_obj["brake"], min_brake, max_break, 0, 1)
    new_obj["steering"] = num_to_range(can_obj["steering"], min_steering, max_steering, -1, 1)
    new_obj["throttle"] = num_to_range(can_obj["throttle"], min_throttle, max_throttle, 0, 1)

    new_obj["brake"] = round(new_obj["brake"], 1)
    new_obj["steering"] = round(new_obj["steering"], 1)
    new_obj["throttle"] = round(new_obj["throttle"], 1)

    return new_obj


class LaneCNN(nn.Module):
    def __init__(self):
        super(LaneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(
            512, 3
        )  # output three values -> [steering, throttle, breaking]

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = x.view(256 * 28 * 28)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


learning_rate = 0.001
batch_size = 64
epochs = 10
model_path = PATH + "/models"

path = Path(model_path)

transform = transforms.Compose(
        [transforms.Resize((224, 224), antialias=True), transforms.ToTensor()]
    )

model_file_name = "nuscenes_model_v1.1.pth"

def train():
    print("Model Version V1.1")
    print("final model weights will be saved to: " + model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    scenes = nusc.scene

    train_size = int(0.8 * len(scenes))
    val_size = len(scenes) - train_size
    train, val = random_split(scenes, [train_size, val_size])

    net = LaneCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print(f"Training on {device}")

    for epoch in range(epochs):
        net.train()
        for scene in train:
            print("Training on scene " + scene['name'])
            scene_number = int(scene['name'].split("-")[1])

            if scene_number in nusc_can.can_blacklist:
                print("Skipping scene " + str(scene_number))
                continue
            
            first_sample_token = scene['first_sample_token']

            current_sample = nusc.get('sample', first_sample_token)

            scene_vehicle_monitor = nusc_can.get_messages(scene['name'], 'vehicle_monitor')

            while True:
                sensor = "CAM_FRONT"
                cam_front_data = nusc.get("sample_data", current_sample["data"][sensor])
                current_image_path = PATH + "/" + cam_front_data["filename"]
                img = Image.open(current_image_path)

                img_input = transform(img).to(device)
            
                current_vehicle_can = get_closest_can(current_sample["timestamp"], scene_vehicle_monitor)                    

                if current_vehicle_can == {}:
                    if current_sample['next'] == '':
                        break
                    else:
                        current_sample = nusc.get('sample', current_sample['next'])
                        continue

                normal_vm_can = normalize_vehicle_monitor_can(current_vehicle_can)

                steering_targets = normal_vm_can['steering']
                throttle_targets = normal_vm_can['throttle']
                breaking_targets = normal_vm_can['brake']

                label = torch.FloatTensor([steering_targets, throttle_targets, breaking_targets]).to(device)              

                optimizer.zero_grad()
                
                # Forward pass
                outputs = net(img_input)

                # Compute loss
                total_loss = criterion(outputs, label)

                # Backward pass
                total_loss.backward()

                # Update weights
                optimizer.step()

                if current_sample['next'] == '':
                    break
                else:
                    current_sample = nusc.get('sample', current_sample['next'])

        # Validation
        net.eval()
        with torch.no_grad():
            for scene in val:
                print("Validating on scene " + scene['name'])

                scene_number = int(scene['name'].split("-")[1])

                if scene_number in nusc_can.can_blacklist:
                    print("Skipping scene " + str(scene_number))
                    continue

                first_sample_token = scene['first_sample_token']

                current_sample = nusc.get('sample', first_sample_token)

                scene_vehicle_monitor = nusc_can.get_messages(scene['name'], 'vehicle_monitor')

                while True:
                    sensor = "CAM_FRONT"
                    cam_front_data = nusc.get("sample_data", current_sample["data"][sensor])
                    current_image_path = PATH + "/" + cam_front_data["filename"]
                    img = Image.open(current_image_path)

                    img_input = transform(img).to(device)

                    current_vehicle_can = get_closest_can(current_sample["timestamp"], scene_vehicle_monitor)

                    if current_vehicle_can == {}:
                        if current_sample['next'] == '':
                            break
                        else:
                            current_sample = nusc.get('sample', current_sample['next'])
                            continue
                    
                    normal_vm_can = normalize_vehicle_monitor_can(current_vehicle_can)

                    steering_targets = normal_vm_can['steering']
                    throttle_targets = normal_vm_can['throttle']
                    breaking_targets = normal_vm_can['brake']

                    label = torch.FloatTensor([steering_targets, throttle_targets, breaking_targets]).to(device)

                    outputs = net(img_input)

                    val_total_loss = criterion(outputs, label)

                    if current_sample['next'] == '':
                        break
                    else:
                        current_sample = nusc.get('sample', current_sample['next'])

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}, Validation Loss: {val_total_loss.item():.4f}"
        )
        torch.save(
            net.state_dict(),
            os.path.join(model_path, "epochs", f"model_e{epoch+1}.pth"),
        )

    print("Finished training")
    torch.save(net.state_dict(), os.path.join(model_path, model_file_name))


if __name__ == "__main__":
    train()
