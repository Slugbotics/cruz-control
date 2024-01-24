#include <DFRobot_BMX160.h>

#include <Servo.h>
#include <Wire.h>
#include <Serial.h>
#include <string>
#define MAX_I2C_MESSAGE 100
#define ESCPin 9
#define ServoPin 3

Servo ESC;
Servo SteeringServo;

DFRobot_BMX160 bmx160;

int esc_value = 1500;  // Set signal value, which should be between 1100 and 1900, 1500 is the center
int servo_deg = 0;

int thrust;
int steering;

bool interrupt = false;
String message = String("");


void setup() {
  Serial.begin(9600);
  SteeringServo.attach(ServoPin);  // Steering servo
  SteeringServo.write(0);  // Set stearing servo to 0 degrees

  ESC.attach(ESCPin);           // Motor ESC
  ESC.writeMicroseconds(1500);  // send "stop" signal to ESC.

  Wire.begin(30);                // By default, pins are GPIO 0 (pin 1) and GPIO 1 (pin 2)
  Wire.onReceive(receiveEvent);  // register event
  Wire.onRequest(requestInterrupt);
  
   if (bmx160.begin() != true){
    Serial.println("init false");
    while(1);
  }          

  delay(3000);  // delay to allow the ESC to recognize the stopped signal
}

void loop() { 
  sBmx160SensorData_t Omagn, Ogyro, Oaccel;

  /* Get a new sensor event */
  bmx160.getAllData(&Omagn, &Ogyro, &Oaccel);

  /* Display the magnetometer results (magn is magnetometer in uTesla) */
  Serial.print("M ");
  Serial.print("X: "); Serial.print(Omagn.x); Serial.print("  ");
  Serial.print("Y: "); Serial.print(Omagn.y); Serial.print("  ");
  Serial.print("Z: "); Serial.print(Omagn.z); Serial.print("  ");
  Serial.println("uT");

  SteeringServo.write(servo_deg);
  ESC.writeMicroseconds(esc_value);  // Send signal to ESC.
}


void requestInterrupt() {

  if (!interrupt && message != "") {
    thrust = message.substring(0, message.indexOf(",")).toInt();
    servo_deg = message.substring(message.indexOf(",")+1, message.length()).toInt();
    message = "";
    Serial.println(thrust+" "+ servo_deg);
  }

}


// Expected string structure: int, int
// First int is the motor percentage, second int is the stearing angle
void receiveEvent(int howMany) {

  char string[MAX_I2C_MESSAGE];
  int recieved = 0;
  while (1 < Wire.available()) {
    char c = Wire.read();
    string[recieved] = c;
    recieved++;
  }

  Serial.println(string);

  int x = Wire.read();  // receive byte as an integer
  string[recieved] = x;
  char* token1 = strtok(string, ", ");
  char* token2 = strtok(NULL, ", ");

  thrust = atoi(token1);
  steering = atoi(token2);

  esc_value = map(thrust, -100, 100, 1100, 1900);
  servo_deg = steering;
}
