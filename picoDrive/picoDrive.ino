// Simple I2C master and slave demo - Earle F. Philhower, III
// Released into the public domain
//
// Using both onboard I2C interfaces, have one master and one slave
// and send data both ways between them
//
// To run, connect GPIO0 to GPIO2, GPIO1 to GPIO3 on a single Pico

#include <Wire.h>

bool data_recieved = false;
static char buff[100];

void setup() {
  Serial.begin(115200);
  Wire1.setSDA(2);
  Wire1.setSCL(3);
  Wire1.begin(0x30);
  Wire1.onReceive(recv);
  Wire1.onRequest(req);
}


void loop() {
  if (data_recieved == true){
    data_recieved = false;
    int m1 = stoi(buff.substr(buff.find(','));
    buff = buff.substr(buff.find(',');
    int m2 = stoi(buff.substr(buff.find(','));
    buff = buff.substr(buff.find(',');
    int ang = stoi(buff.substr(buff.find(','));
    buff = buff.substr(buff.find(',');
  }
}

// These are called in an **INTERRUPT CONTEXT** which means NO serial port
// access (i.e. Serial.print is illegal) and no memory allocations, etc.

// Called when the I2C slave gets written to
void recv(int len) {
  buff[bytes_recieved]=Wire1.read();
 
  if (character=="\n"){
    data_recieved = true;
    bytes_recieved=0;
  }
  
  bytes_recieved++
}

// Called when the I2C slave is read from
void req() {
  static int ctr = 765;
  char buff[7];
  // Return a simple incrementing hex value
  sprintf(buff, "%06X", (ctr++) % 65535);
  Wire1.write(buff, 6);
}
