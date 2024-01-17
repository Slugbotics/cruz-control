#include <Servo.h>

Servo steering;  // create servo object to control a servo

int direction = 0;    //direction of servo. on uno it is 0-180 with 90 being center, plan to have the pi send -90 to 90

void setup() {
  Serial.begin(9600);
  steering.attach(9);  // attaches the servo on pin 9 to the servo object
  steering.write(90);  // set servo to mid-point
}

void loop() {
  //read i2c data from pi, add 90 to pi
  
}