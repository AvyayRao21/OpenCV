   const int redPin = 9;    // Pin connected to the red LED
   const int greenPin = 10; // Pin connected to the green LED
   const int bluePin = 11;  // Pin connected to the blue LED

   void setup() {
       Serial.begin(9600);
       pinMode(redPin, OUTPUT);
       pinMode(greenPin, OUTPUT);
       pinMode(bluePin, OUTPUT);
   }

   void loop() {
       if (Serial.available() > 0) {
           String rgbString = Serial.readStringUntil('\n');
           int red, green, blue;
           sscanf(rgbString.c_str(), "%d,%d,%d", &red, &green, &blue);
           analogWrite(redPin, red);
           analogWrite(greenPin, green);
           analogWrite(bluePin, blue);
       }
   }
   