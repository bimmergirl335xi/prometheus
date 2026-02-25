// octapod_servos.ino (Runs on Arduino Giga)
#include <Servo.h>

// For the proof-of-concept, we map the 8 angles to 8 base servos
const int NUM_LEGS = 8;
Servo legs[NUM_LEGS];

// Assign the GPIO pins you are using on the Giga
int servoPins[NUM_LEGS] = {2, 3, 4, 5, 6, 7, 8, 9};

// A "Union" allows us to read raw bytes from the serial port 
// and instantly treat them as a Float array without any slow math conversions.
union FloatArray {
    byte bytes[32];      // 8 floats * 4 bytes each = 32 bytes
    float angles[NUM_LEGS]; 
} motorData;

void setup() {
    // Must match the baud rate in the Pi python script
    Serial.begin(115200);
    
    // Attach the servos to the hardware timers
    for(int i = 0; i < NUM_LEGS; i++) {
        legs[i].attach(servoPins[i]);
        
        // Move to neutral standing position (90 degrees) on boot
        legs[i].write(90); 
    }
}

void loop() {
    // If the Pi 5 has sent a full 32-byte packet
    if (Serial.available() >= 32) {
        
        // Read the bytes directly into the union
        Serial.readBytes(motorData.bytes, 32);
        
        // Apply the angles to the servos
        for(int i = 0; i < NUM_LEGS; i++) {
            // The Julia server sends sine waves between -1.0 and 1.0.
            // We map that math to physical servo degrees (0 to 180).
            // (Multiply by 100 to map properly with integers)
            int targetDegree = map(motorData.angles[i] * 100, -100, 100, 0, 180);
            
            // Constrain to prevent the servos from ripping themselves apart
            targetDegree = constrain(targetDegree, 10, 170);
            
            legs[i].write(targetDegree);
        }
    }
}