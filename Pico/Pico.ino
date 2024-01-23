#include <Servo.h>
#include <Wire.h>
#define MAX_I2C_MESSAGE 100
#define ESCPin = 2;
#define ServoPin = 3;

Servo ESC;
Servo Servo;

int esc_value = 1500;  // Set signal value, which should be between 1100 and 1900, 1500 is the center
int servo_deg = 0;

void setup() {
  Servo.attach(ServoPin);  // Stearing servo
  Servo.write(0);          // Set stearing servo to 0 degrees

  ESC.attach(ESCPin);           // Motor ESC
  ESC.writeMicroseconds(1500);  // send "stop" signal to ESC.

  Wire.begin(30);                // By default, pins are GPIO 0 (pin 1) and GPIO 1 (pin 2)
  Wire.onReceive(receiveEvent);  // register event
  Serial.begin(9600);            // start serial for output

  delay(7000);  // delay to allow the ESC to recognize the stopped signal
}

void loop() {
  Servo.write(servo_deg);
  ESC.writeMicroseconds(esc_value);  // Send signal to ESC.
}

// Expected string structure: int, int
// First int is the motor percentage, second int is the stearing angle
void receiveEvent(int howMany) {
  char* string[MAX_I2C_MESSAGE];
  int recieved = 0;
  while (1 < Wire.available()) {
    char c = Wire.read();
    string[recieved] = c;
    recieved++;
  }
  int x = Wire.read();  // receive byte as an integer
  string[recieved] = x;
  char* token1 = strtok(string, ", ");
  char* token2 = strtok(NULL, ", ");

  int thrust = atoi(token1);
  int stearing = atoi(token2);

  esc_value = map(val, -100, 100, 1100, 1900);
  servo_deg = stearing;
}
