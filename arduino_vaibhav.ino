#include <Servo.h>

Servo servoPan;
Servo servoTilt;
int ledPin = 8; 

void setup() {
  Serial.begin(9600);
  servoPan.attach(9);
  servoTilt.attach(10);
  pinMode(ledPin, OUTPUT);

  servoPan.write(90);
  servoTilt.write(90);
  digitalWrite(ledPin, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    
    int pan = Serial.parseInt();
    int tilt = Serial.parseInt();

    pan = constrain(pan, 0, 180);
    tilt = constrain(tilt, 0, 180);

    servoPan.write(pan);
    servoTilt.write(tilt);
    int ledState = Serial.parseInt();
    digitalWrite(ledPin, ledState);

    while (Serial.available()) Serial.read();
  }
}
