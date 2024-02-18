from smbus import SMBus
import pygame
pygame.init()

import os
os.environ['SDL_JOYSTICK_HIDAPI_XBOX'] = '1'
os.environ['SDL_JOYSTICK_HIDAPI_XBOX_ONE'] = '1'
Controller = pygame.joystick.Joystick(0)
Controller.init()
print ("Detected joystick " + Controller.get_name())


triggers = [-1.0,-1.0] #left joystick, right joystick [-1 default, 1 pressed]
joystick1 = [0,0] #x axis [-1 left, 1 right] y axis [-1 up, 1 down]
joystick2 = [0,0] #x axis [-1 left, 1 right] y axis [-1 up, 1 down]
dpad = [0,0] #x axis [-1 left, 1 right] y axis [-1 down, 1 up]
buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # a b x y lb rb share option lhat rhat

maxSteer = 50
maxThrottle = 20

prev_angle = 0
prev_target = 0

def num_to_range(num, inMin, inMax, outMin, outMax):
  return int(outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin)))

addr = 0x8 # Pico communication bus on address 4

def connect(port):
	global bus
	bus = SMBus(port)

def string2Bytes(val):
	retVal = []
	for c in val:
		retVal.append(ord(c))
	return retVal

def send(data):
	bus.write_i2c_block_data(addr, 0x00, data) 
	return -1

def recv():
     pass

connect(1)

while True:
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 4:
                triggers[0]=event.value + 1
            if event.axis == 5:
                triggers[1]=event.value + 1
            if event.axis == 0:
                joystick1[0]=event.value
            if event.axis == 1:
                joystick1[1]=event.value
            if event.axis == 2:
                joystick2[0]=event.value
            if event.axis == 3:
                joystick2[1]=event.value
        if event.type == pygame.JOYHATMOTION: #d-pad
            dpad=[event.value[0],event.value[1]]
        if event.type == pygame.JOYBUTTONUP:
            buttons[event.button]=0
        if event.type == pygame.JOYBUTTONDOWN:
            buttons[event.button]=1
            
        #on pi, triggers[0] and [1] is flipped, windows order, triggers[1]-triggers[0]
        combinedTriggerValue = triggers[0]/2 - triggers[1]/2

        if combinedTriggerValue > 1:
             combinedTriggerValue = 1
        if combinedTriggerValue < -1:
             combinedTriggerValue = -1

        target = num_to_range(combinedTriggerValue, -1, 1, -maxThrottle, maxThrottle)
        angle = num_to_range(joystick1[0], -1, 1, -maxSteer, maxSteer)
        

        sendString =  str(angle) + "," + str(target)

        print(sendString)
	
        if angle < 0:
            if target < 0:
                send([1,abs(angle) ,1, abs(target)])
            else:
                send([1, abs(angle), 0, abs(target)])
        else:
            if target < 0:
                send([0, abs(angle),1, abs(target)])
            else:
                send([0, abs(angle), 0, abs(target)])

