import gymnasium as gym
from stable_baselines3 import PPO
import torch
import os
import numpy as np
import cv2

device = "cuda"
model = PPO.load("../ppo-racecar.zip")

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    img = np.transpose(img, (2, 0, 1))
    action, _ = model.predict(img)
    print(action)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break