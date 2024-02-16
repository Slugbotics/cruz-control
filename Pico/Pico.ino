#include <Servo.h>
#include <Wire.h>

#define ServoPin 3
#define ESCPin 9

Servo SteeringServo;
Servo ESC;

int servo_freq = 1500;
int thrust_freq = 1500;

char c;
int a_neg;
int angle;
int t_neg;
int thrust_p;

bool interrupt = false;

void setup() {
  SteeringServo.attach(ServoPin);
  SteeringServo.writeMicroseconds(servo_freq);

  ESC.attach(ESCPin);
  ESC.writeMicroseconds(thrust_freq);

  Wire.begin(0x8);
  Wire.onReceive(receiveEvent);

  delay(3000);
}

void loop() {
  SteeringServo.writeMicroseconds(servo_freq);
  ESC.writeMicroseconds(thrust_freq);
}

void receiveEvent(int count) {
  c = Wire.read();
  a_neg = Wire.read();
  angle = Wire.read();
  t_neg = Wire.read();
  thrust_p = Wire.read();

  if (a_neg == 1) {
    angle = angle * -1;
  }

  if (t_neg == 1) {
    thrust_p = thrust_p * -1;
  }

  servo_freq = map(angle, -50, 50, 1100, 1900);
  thrust_freq = map(thrust_p, -100, 100, 1100, 1900);

  Serial.print(servo_freq);
  Serial.print(",");
  Serial.print(thrust_freq);
  Serial.print("\n");
}
