# This file needs work
import carla
import math
import time
import random
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import os
import numpy as np
import cv2

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
camera_init_trans = carla.Transform(carla.Location(z=1.5, x=2))

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
device = "cpu"
model = PPO.load("ppo-racecar.zip") # Set this path appropriately
control = carla.VehicleControl()
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE) # Visualize camera
cv2.imshow('RGB Camera', sensor_data['image'])
cv2.waitKey(1)

while(True):
    cv2.imshow('RGB Camera', sensor_data['image'])
    frame = sensor_data['image']
    img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # Our operations on the frame come here
    img = cv2.resize(img, (128,128))
    img = np.transpose(img, (2, 0, 1))
    action, _ = model.predict(img)
    print(action)
    control.throttle = float(action[0])
    control.steer = float(action[1])
    ego_vehicle.apply_control(control)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()