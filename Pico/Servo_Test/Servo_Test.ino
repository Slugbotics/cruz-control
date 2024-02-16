#include <Servo.h>
#include <Wire.h>
#include <Serial.h>
#include <string>
#define MAX_I2C_MESSAGE 100
#define ESCPin 9
#define ServoPin 3

Servo ESC;
Servo SteeringServo;

void setup() {
  Serial.begin(9600);
  SteeringServo.attach(ServoPin);  // Steering servo
  SteeringServo.write(90);  // Set stearing servo to 0 degrees
}

void loop() {
  // put your main code here, to run repeatedly:

}
