#include <Servo.h>
#include <Wire.h>

#define ServoPin 3                                              // GPIO Pin used to control the Servo using PWM
#define ESCPin 9                                                // GPIO Pin used to control the ESC (Electronic Speed Controler) using PWM
#define MIN_FREQ 1100                                           // Lower bound frequency for PWM
#define MAX_FREQ 1900                                           // Upper bound frequency for PWM

#define I2C_SLAVE_ADDRESS 0x8                                   // I2C Address that the master will see

Servo SteeringServo;                                
Servo ESC;

int center_point = 1500;

int servo_freq = center_point;                                  // Steering servo Variable
int thrust_freq = center_point;                                 // ESC Value 

void setup() {
  SteeringServo.attach(ServoPin);                               // Set Servo pin to steering servo object
  SteeringServo.writeMicroseconds(servo_freq);                  // Center the steering servo

  ESC.attach(ESCPin);                                           // Set the ESC pin to the ESC object
  ESC.writeMicroseconds(thrust_freq);                           // Set the motor to 0

  Wire.begin(I2C_SLAVE_ADDRESS);                                                 // Initialize I2C on specified address
  Wire.onReceive(receiveEvent);

  delay(3000);                                                  // Wait for 3 seconds, this is because the ESC must recieve the 0 command on boot
}

void loop() {
  SteeringServo.writeMicroseconds(servo_freq);                  // Set angle of steering servo
  ESC.writeMicroseconds(thrust_freq);                           // Set speed of ESC
  delay(100);                                                   // Delay added to ensure the I2C buffer isn't too large
}

void receiveEvent(int count) {
  char c = Wire.read();                                         // Command Byte
  int a_neg = Wire.read();                                      // Is Angle Negative, ie. turn direction
  int angle = Wire.read();                                      // Turning Angle
  int t_neg = Wire.read();                                      // Is Thrust negative
  int thrust_p = Wire.read();                                   // Thrust percentage

  if (a_neg == 1) {
    angle = angle * -1;
  }

  if (t_neg == 1) {
    thrust_p = thrust_p * -1;
  }

  servo_freq = map(angle, -50, 50, MIN_FREQ, MAX_FREQ);         // Set steering servo frequency
  thrust_freq = map(thrust_p, -100, 100, MIN_FREQ, MAX_FREQ);   // Set ESC frequency
}
