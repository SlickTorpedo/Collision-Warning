#include <Servo.h>

// Servo object
Servo myservo;

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

  // Move servo to default position (optional)
  myservo.write(map(0, -90, 90, 0, 180));
  delay(500); // Allow servo to stabilize
}

void loop() {
  int angle;

  // Check if serial data is available
  if (Serial.available() > 0) {
    // Read the angle input as a string and trim any non-numeric characters
    String input = Serial.readStringUntil('\n');
    input.trim(); // Remove leading/trailing whitespace

    // Parse the input as an integer
    angle = input.toInt();

    // Ensure the angle is within valid range
    if (angle < -90) angle = -90;
    if (angle > 90) angle = 90;

    // Move the servo to the specified angle
    myservo.write(map(angle, -90, 90, 0, 180));
    
    // Wait for the servo to stabilize at the new position
    delay(10); // Adjust this delay if needed for your servo

    // Measure LIDAR distance
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
  }
}