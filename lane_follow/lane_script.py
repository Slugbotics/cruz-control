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

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

blueprint_library = world.get_blueprint_library()

bp = random.choice(blueprint_library.filter('vehicle'))

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
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Set spectator transform, allow time for spawning
time.sleep(0.2)
spectator.set_transform(camera.get_transform())
camera.destroy()
# # Model Control

# device = "cpu"
# model = PPO.load("../ppo-racecar.zip")
# cc = carla.ColorConverter.Raw
# control = carla.VehicleControl()
# while(True):
#     # Capture frame-by-frame
#     camera.listen(lambda image: image.save_to_disk('out/frame.png', cc)) # This does not seem to be generating the image file at the moment, work on this

#     frame = cv2.imread('_out/frame.png') 
#     # Our operations on the frame come here
#     img = cv2.resize(frame, (128,128))
#     img = np.transpose(img, (2, 0, 1))
#     action, _ = model.predict(img)
#     control.throttle = action[0]
#     control.steer = action[1]
#     ego_vehicle.apply_control(control)