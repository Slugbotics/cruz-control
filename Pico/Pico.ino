#include <Servo.h>
#include <Wire.h>
// #include <Serial.h>
#include <string>
#define MAX_I2C_MESSAGE 100
#define ESCPin 9
#define ServoPin 3

Servo ESC;
Servo SteeringServo;


int esc_value = 1550;  // Set signal value, which should be between 1100 and 1900, 1500 is the center
int servo_deg = 90;

int thrust = 1500; // Set signal value, which should be between 1100 and 1900, 1500 is the stop command
int steering;

bool interrupt = false;

char inpData[32]; // 32 byte char array, can be larger based on data needs
String data = String();



void setup() {
  Serial.begin(9600);
  SteeringServo.attach(ServoPin);  // Steering servo
  SteeringServo.write(90);  // Set stearing servo to 0 degrees

  ESC.attach(ESCPin);           // Motor ESC
  ESC.writeMicroseconds(1500);  // send "stop" signal to ESC.

  Wire.begin(0x8);               // communicate on address 8     
  Wire.onReceive(receiveEvent);  // register events
  Wire.onRequest(requestInterrupt);
  
  delay(3000);  // delay to allow the ESC to recognize the stopped signal
}

void loop() { 
  if (interrupt == true) {
    Serial.print("Thrust: ");
    Serial.print(thrust);
    Serial.print(" Steering: ");
    Serial.println(steering);
    interrupt = false;
  }


  delay(100);
  

  SteeringServo.write(servo_deg);
  ESC.writeMicroseconds(thrust);  // Send signal to ESC.
}


void requestInterrupt() {

  // if (!interrupt && message != "") {
  //   thrust = message.substring(0, message.indexOf(",")).toInt();
  //   servo_deg = message.substring(message.indexOf(",")+1, message.length()).toInt();
  //   message = "";
  //   Serial.println(thrust+" "+ servo_deg);
  // }

}


void receiveEvent(int howMany) {

  for (int i = 0; i < howMany; i++) {
    inpData[i] = Wire.read();
    inpData[i + 1] = '\0'; //add null after ea. char
  }

  //RPi first byte is cmd byte so shift everything to the left 1 pos so temp contains our string
  for (int i = 0; i < howMany; ++i) inpData[i] = inpData[i + 1];
  data = inpData;
  thrust = map(data.substring(0, data.indexOf(",")).toInt(),-100,100,1100,1900);
  servo_deg = map(data.substring(data.indexOf(",")+1, data.length()).toInt(), -15,15, 55,125);
}


// void receiveEvent(int howMany) {
//   Serial.println("recv");

//   char string[MAX_I2C_MESSAGE];
//   int recieved = 0;
//   while (1 < Wire.available()) {
//     char c = Wire.read();
//     string[recieved] = c;
//     recieved++;
//   }

//   Serial.println(string);

//   int x = Wire.read();  // receive byte as an integer
//   string[recieved] = x;
//   char* token1 = strtok(string, ", ");
//   char* token2 = strtok(NULL, ", ");

//   thrust = atoi(token1);
//   steering = atoi(token2);

//   esc_value = map(thrust, -100, 100, 1100, 1900);
//   servo_deg = steering;
// }
