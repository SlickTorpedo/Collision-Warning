import serial
import matplotlib.pyplot as plt
import math

# Function to connect to the specified serial port
def find_arduino():
    print("Connecting to Arduino on /dev/cu.usbserial-2110...")
    try:
        ser = serial.Serial('/dev/cu.usbserial-2110', 115200, timeout=1)
        return ser
    except Exception as e:
        raise Exception(f"Failed to connect to Arduino: {e}")
    
def coordinates_to_angle(x, y):
    angle = math.degrees(math.atan2(y, x))
    angle = angle - 90
    return angle

# Connect to Arduino
try:
    arduino = find_arduino()
except Exception as e:
    print(f"Error: {e}")
    exit()

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_title("LIDAR Sensor Mapping")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_xlim(-300, 300)  # Adjusted for a centered view
ax.set_ylim(0, 500)    # Adjusted for typical LIDAR range

try:
    dummy_starter_data_sent = False #For some reason you have to send something because the first one always fails
    while True:
        # Prompt user for an angle
        if dummy_starter_data_sent:
            angle_input = input("Enter an angle (-90 to 90): ")
        else:
            angle_input = "0"
            dummy_starter_data_sent = True
        try:
            angle = int(angle_input)
            if angle < -90 or angle > 90:
                print("Please enter a valid angle between -90 and 90.")
                continue
        except ValueError:
            print("Invalid input. Please enter a numeric angle.")
            continue

        # Send the angle to the Arduino
        arduino.write(f"{angle}\n".encode())

        # Wait for response
        line = arduino.readline().decode('utf-8').strip()
        if line and "Angle" in line:
            print(f"Received data: {line}")
            try:
                # Parse angle and distance
                angle_str, distance_str = line.split(", Distance: ")
                angle = int(angle_str.split(": ")[1].replace("°", ""))
                distance = distance_str.split(" ")[0]

                if distance == "ERROR":
                    print(f"Invalid reading received at angle {angle}")
                    continue

                distance = float(distance)

                # Convert polar coordinates to Cartesian
                x = distance * math.cos(math.radians(angle + 90))
                x = -x #Flip the X axis
                y = distance * math.sin(math.radians(angle + 90))

                print("Calculated angle ->", coordinates_to_angle(x, y))

                # Clear previous points and plot the new point
                sc.set_offsets([(x, y)])
                plt.pause(0.01)
                print(f"Plotted point -> X: {x:.2f}, Y: {y:.2f}, Distance: {distance:.2f} cm")
            except ValueError as ve:
                print(f"Error parsing data: {line} -> {ve}")
        else:
            print("No valid data received from Arduino.")
except KeyboardInterrupt:
    print("\nExiting program on user interruption...")
    arduino.close()
finally:
    print("Closing serial port and plot.")
    plt.ioff()
    plt.show()