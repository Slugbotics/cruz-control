from gpiozero import AngularServo
from time import sleep

servo = AngularServo(17)

while (True):
    servo.angle = 0
    sleep(2)