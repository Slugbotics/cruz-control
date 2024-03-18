import glob
import os
import sys
import carla
import torch
from torch import nn
from torchvision import transforms
import random
import time
import numpy as np
import cv2
import PIL.Image as Image

if len(sys.argv) != 2:
    print("Usage: python lane_script.py <model_path>")
    sys.exit(1)

IM_WIDTH = 1000
IM_HEIGHT = 1000

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

# Model Control
device = torch.device("cuda")
model = LaneCNN().to(device)
model.load_state_dict(torch.load(sys.argv[1], map_location=device))
control = carla.VehicleControl()

sensor_data = {"image": None}

def process_img(image, sensor_data):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    sensor_data["image"] = i3


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    print("Loading")

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    sensor.listen(lambda data: process_img(data, sensor_data))

    while True:
        if sensor_data["image"] is not None:
            image_raw = sensor_data["image"]
            cv2.imshow("", image_raw)
            cv2.waitKey(1)
            image = Image.fromarray(image_raw)
            img_transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
            input_image = img_transform(image).to(device)
            prediction = model(input_image)
            prediction = torch.round(prediction, decimals=1)
            print(prediction)
            control.steer = prediction[0].item()
            control.throttle = prediction[1].item() * 2
            # control.brake = prediction[2].item()   # This line currently makes the vehicle unable to move
            vehicle.apply_control(control)
            world.tick()


finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')