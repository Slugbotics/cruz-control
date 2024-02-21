from gpiozero import Servo
from enum import Enum
import pygame

# initialize pygame for services like joystick detection
pygame.init()

# initialize controller under pygame
Controller = pygame.joystick.Joystick(0)
Controller.init()
print ("Detected joystick " + Controller.get_name())

# values to store joystick state
triggers = [-1.0,-1.0] #left joystick, right joystick [-1 default, 1 pressed]
joystick1 = [0,0] #x axis [-1 left, 1 right] y axis [-1 up, 1 down]
joystick2 = [0,0] #x axis [-1 left, 1 right] y axis [-1 up, 1 down]
dpad = [0,0] #x axis [-1 left, 1 right] y axis [-1 down, 1 up]
buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # a b x y lb rb share option lhat rhat

maxSteer = 80 # max angle (percent)
maxThrottle = 20 # max throttle (percent)

# values to write to servos
angle = 0 
throttle = 0

# initialize servo objects on gpio 11, 12
SteeringServo = Servo(11) 
DrivingServo = Servo(12)

# function to round num to range, returns 2 decimal point value
def num_to_range(num, inMin, inMax, outMin, outMax):
  return round(outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin)), 2)

# enum mappings for xbox controller readability
# triggers and joysticks are registered under the axis event, use enum for clarity
class XB_AXIS_MAP(int, Enum):
     LTRIGGER=4
     RTRIGGER=5
     LJOYX=0
     LJOYY=1
     RJOYX=2
     RJOYY=3

# main program loop
while True:
    SteeringServo.value = angle
    DrivingServo.value = throttle
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == XB_AXIS_MAP.LTRIGGER:
                triggers[0]=event.value + 1
            if event.axis == XB_AXIS_MAP.RTRIGGER:
                triggers[1]=event.value + 1
            if event.axis == XB_AXIS_MAP.LJOYX:
                joystick1[0]=event.value
            if event.axis == XB_AXIS_MAP.LJOYY:
                joystick1[1]=event.value
            if event.axis == XB_AXIS_MAP.RJOYX:
                joystick2[0]=event.value
            if event.axis == XB_AXIS_MAP.RJOYY:
                joystick2[1]=event.value
        if event.type == pygame.JOYHATMOTION: #d-pad
            dpad=[event.value[0],event.value[1]]
        if event.type == pygame.JOYBUTTONUP:
            buttons[event.button]=0
        if event.type == pygame.JOYBUTTONDOWN:
            buttons[event.button]=1
            
        #on pi, triggers[0] and [1] is flipped, windows order, triggers[1]-triggers[0]
        combinedTriggerValue = triggers[0]/2 - triggers[1]/2

        # servo library breaks with values over expected range of -1, 1
        # ranges of angle and throttle are scaled to max steering angle 
        throttle = num_to_range(combinedTriggerValue, -1, 1, -maxThrottle/100, maxThrottle/100)
        angle = num_to_range(joystick1[0], -1, 1, -maxSteer/100, maxSteer/100)
        
        # print values
        print(str(throttle)+" "+str(angle))
        
