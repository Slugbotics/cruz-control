import carla
import math
import time
import random
import gymnasium as gym
from stable_baselines3 import PPO
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2

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
        self.fc2 = nn.Linear(512, 3) # output three values -> [steering, throttle, breaking]

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = x.view(256* 28*28)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    

def rgb_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Connect to the client and retrieve the world object

client = carla.Client('localhost', 2000)
world = client.get_world()

blueprint_library = world.get_blueprint_library()

bp = blueprint_library.find('vehicle.dodge.charger_2020')

# A blueprint contains the list of attributes that define a vehicle's
# instance, we can read them and modify some of them. For instance,
# let's randomize its color.
if bp.has_attribute('color'):
    color = random.choice(bp.get_attribute('color').recommended_values)
    bp.set_attribute('color', color)

# Now we need to give an initial transform to the vehicle. We choose a
# random transform from the list of recommended spawn points of the map.
transform = random.choice(world.get_map().get_spawn_points())

# So let's tell the world to spawn the vehicle.
ego_vehicle = world.spawn_actor(bp, transform)

# Create a transform for the spectator
spectator = world.get_spectator()
spec_trans = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), ego_vehicle.get_transform().rotation)
spectator.set_transform(spec_trans)

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=0.5, x=5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Set spectator transform, allow time for spawning
time.sleep(0.2)
# Initialize sensor data dict

image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

sensor_data = {'image': np.zeros((image_h, image_w, 3))}

camera.listen(lambda image: rgb_callback(image, sensor_data))

# Model Control
device = torch.device("cuda")
model = LaneCNN().to(device)
model.load_state_dict(torch.load("model.pth"))
control = carla.VehicleControl()
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE) # Visualize camera
cv2.imshow('RGB Camera', sensor_data['image'])
cv2.waitKey(1)

while(True):
    cv2.imshow('RGB Camera', sensor_data['image'])
    frame = sensor_data['image']
    img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).astype(np.float32)
    img = cv2.resize(img, (224,224))
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).to(device)
    # Our operations on the frame come here
    prediction = model(tensor)
    print(prediction)
    control.steer = prediction[0].item()
    control.throttle = prediction[1].item()
    control.brake = prediction[2].item()   # This line currently makes the vehicle unable to move
    ego_vehicle.apply_control(control)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
camera.destroy()
ego_vehicle.destroy()