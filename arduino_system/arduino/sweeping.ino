//This program will just sweep and give you back all the data. The servo is kinda loud and I don't want to have it sweep the whole time so I am making a dynamic version that looks only where the road is.
#include <Servo.h>

// Servo object
Servo myservo;

// Variables for servo motion
int angle = -90;     // Starting angle
int stepSize = 1;   // Step size for sweeping

// Variables for LIDAR PWM
unsigned long pulseWidth;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  // Setup pins for LIDAR
  pinMode(2, OUTPUT);  // Set pin 2 as trigger pin
  digitalWrite(2, LOW); // Set trigger LOW for continuous read
  pinMode(3, INPUT);   // Set pin 3 as monitor pin

  // Initialize servo
  myservo.attach(9); // Attach the servo to pin 9
  Serial.println("Servo and LIDAR Initialized");

  // Move servo to starting position
  myservo.write(map(angle, -90, 90, 0, 180));
  delay(500); // Allow servo to stabilize
}

void loop() {
  // Move servo to the current angle
  myservo.write(map(angle, -90, 90, 0, 180));
  delay(10); // Allow servo to reach position

  // Read LIDAR distance
  pulseWidth = pulseIn(3, HIGH); // Measure the pulse duration
  if (pulseWidth != 0) {
    pulseWidth = pulseWidth / 10; // Convert pulse duration to distance in cm
    Serial.print("Angle: ");
    Serial.print(angle);
    Serial.print("°, Distance: ");
    Serial.print(pulseWidth);
    Serial.println(" cm");
  } else {
    Serial.print("Angle: ");
    Serial.print(angle);
    Serial.println("°, Distance: ERROR");
  }

  // Update servo angle
  angle += stepSize;

  // Reverse direction at limits
  if (angle > 90 || angle < -90) {
    stepSize = -stepSize; // Change direction
    angle += stepSize;    // Correct overshoot
  }

  // Short delay between sweeps
  delay(5);
}