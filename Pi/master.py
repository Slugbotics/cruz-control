import os
import time
from datetime import datetime
import csv
from enum import Enum
from smbus import SMBus
import pygame
import cv2
from BMX160 import BMX160
pygame.init()

# check if controller detected
if pygame.joystick.get_count() == 0:
	raise Exception("No gamepads detected!")

# commands to enable xbox controller vibration motors
os.environ['SDL_JOYSTICK_HIDAPI_XBOX'] = '1'
os.environ['SDL_JOYSTICK_HIDAPI_XBOX_ONE'] = '1'
Controller = pygame.joystick.Joystick(0)
Controller.init()

# print statement to notify that controller is detected
print ("Detected joystick " + Controller.get_name())

# arrays to store joystick inputs, triggers, dpad, buttons, and each joystick is treated as a separate entity in pygame
# 
# PYGAME MAPPINGS
#   left joystick, right joystick [-1 default, 1 pressed]
#   x axis [-1 left, 1 right] y axis [-1 up, 1 down]
#   x axis [-1 left, 1 right] y axis [-1 up, 1 down]
#   x axis [-1 left, 1 right] y axis [-1 down, 1 up]
#   a, b, x, y, lb, rb, share, option, lhat, rhat [0 unpressed, 1 [ressed]]
triggers = [-1.0,-1.0]
joystick1 = [0,0] 
joystick2 = [0,0] 
dpad = [0,0] 
buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 

# max steering and throttle range, goes from -max to max for each variable
maxSteer = 50
maxThrottle = 5

# store thrust, angle value being sent by pi for recording
targetString, cameraData = "NO-HEADING-DATA", "NO-CAMERA-DATA"

# triggers and joysticks are registered under the axis event, use enum for clarity
class XB_AXIS_MAP(Enum):
     LTRIGGER=4
     RTRIGGER=5
     LJOYX=0
     LJOYY=1
     RJOYX=2
     RJOYY=3

# map number to range
def num_to_range(num, inMin, inMax, outMin, outMax):

  # get difference between given value and minimum, divide by minimum range, then multiply by maximum range
  # round to 2 decimals
  return round(outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin)), 2)

# Pico communication bus on address 4
addr = 0x8

# intialize SMBus to port
def connect(port):
	global bus
	bus = SMBus(port)

# convert string to bytes
def string2Bytes(val):
	retVal = []
	for c in val:
		retVal.append(ord(c))
	return retVal

# send data over i2c
def send(data):
	byteValue = string2Bytes(data)
	bus.write_i2c_block_data(addr, 0x00, byteValue) 
	return -1

def recv():
     pass

def controllerLoop():

    # read through all events
    for event in pygame.event.get():

        # assign all event values to respective arrays
        # axis event (joystick and triggers)
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == XB_AXIS_MAP.LTRIGGER:
                triggers[0]=event.value
            if event.axis == XB_AXIS_MAP.RTRIGGER:
                triggers[1]=event.value
            if event.axis == XB_AXIS_MAP.LJOYX:
                joystick1[0]=event.value
            if event.axis == XB_AXIS_MAP.LJOYY:
                joystick1[1]=event.value
            if event.axis == XB_AXIS_MAP.RJOYX:
                joystick2[0]=event.value
            if event.axis == XB_AXIS_MAP.RJOYY:
                joystick2[1]=event.value
        
        # joyhatmotion event (dpad)
        if event.type == pygame.JOYHATMOTION:
            dpad=[event.value[0],event.value[1]]
        
        # joybutton events (buttons)
        if event.type == pygame.JOYBUTTONUP:
            buttons[event.button] = 0
        if event.type == pygame.JOYBUTTONDOWN:
            buttons[event.button] = 1
            
        # on pi, triggers[0] and [1] is flipped, windows order, triggers[1]-triggers[0]
        # set combinedTriggerValue to be +-1 if one trigger is pressed, 0 if both or none is pressed 
        combinedTriggerValue = (triggers[0] + 1)/2 - (triggers[1] + 1)/2

        # small bug makes triggers stick to value, cap total output to be in desired range
        if combinedTriggerValue > 1:
             combinedTriggerValue = 1
        if combinedTriggerValue < -1:
             combinedTriggerValue = -1

        # map trigger input and joystick input to maxThrottle and maxSteer range
        target = num_to_range(combinedTriggerValue, -1, 1, -maxThrottle, maxThrottle)
        angle = num_to_range(joystick1[0], -1, 1, -maxSteer, maxSteer)

        # create string to send over i2c
        sendString =  str(angle) + "," + str(target)

        # return string
        return sendString





# record timestamp and get date-time in YYYY-MM-DD HH:MM:SS
def getTime():
    timestamp = datetime.now().timestamp()
    time = str(datetime.fromtimestamp(timestamp))

    # format the time string, removing miliseconds, periods, replacing colons
    # windows doesn't allow for colons in filenames (hiddenstreams) use Modifier Letter Colon instead
    time = time[:time.find('.')]
    return str(time)

# set name of file to time it was recorded
fileName = "data"

# createcsv file and video
file_directory = str(os.path.dirname(__file__))
output_path = file_directory + "/csv_out/"
csv_out = open(output_path + str(fileName) + ".csv", "w")
csvwriter = csv.DictWriter(csv_out)

video = cv2.VideoCapture(0) 

# check if camera is open
if (video.isOpened() == False):  
    raise Exception("Camera is opened by another program") 

# get camera resolution
frame_width = int(video.get(3)) 
frame_height = int(video.get(4))  
size = (frame_width, frame_height) 

# create avi file
video_file = cv2.VideoWriter(file_directory + str(fileName) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 

# csv format
# timestamp, thrust, steering angle, acceleration
def record():
    # get data from bmx as array
    # ARRAY ORDER
    # magn: x, y, z, gyro: x, y, z, accel: x, y, z
    imuData= bmx.get_all_data()
    
    # print data
    # print("magn: x: {0:.2f} uT, y: {1:.2f} uT, z: {2:.2f} uT".format(data[0],data[1],data[2]))
    # print("gyro  x: {0:.2f} g, y: {1:.2f} g, z: {2:.2f} g".format(data[3],data[4],data[5]))
    # print("accel x: {0:.2f} m/s^2, y: {1:.2f} m/s^2, z: {2:.2f} m/s^2".format(data[6],data[7],data[8]))


    # create dictionary to write
    csvData = {"timestamp: ": getTime(),
               "thrust-steering: ": targetString,
               "imu: ": imuData}
    
    # write to csv file using csvWriter
    csvwriter.writerow(csvData)

if __name__ == "__main__":

    bmx = BMX160(1)
    Camera = VideoCapture()
    connect(1)

    # wait for bmx to finish initializing
    while not bmx.begin():
        time.sleep(2)
        
    # loop through controller function and send target values over I2C
    while True:
        targetString = controllerLoop()
        # cameraData = Camera.read()
        
        ret, frame = video.read() 
        if ret == True:
            video_file.write(frame)
        
        record()
        send(targetString)

    # Release objects and close frames
    # video.release() 
    # result.release() 
    # cv2.destroyAllWindows() 
