import serial
import matplotlib.pyplot as plt
import math

# Function to connect to the specified serial port
def find_arduino():
    print("Connecting to Arduino on /dev/cu.usbserial-110...")
    try:
        ser = serial.Serial('/dev/cu.usbserial-110', 115200, timeout=1)
        return ser
    except Exception as e:
        raise Exception(f"Failed to connect to Arduino: {e}")

# Connect to Arduino
try:
    arduino = find_arduino()
except Exception as e:
    print(f"Error: {e}")
    exit()

# Initialize data storage for plotting
points = {}

DO_NOT_PLOT_ZEROS = True
MIN_RANGE_PLOT = 20 #Anything less than 10cm will not be plotted
FLIP_X = True

PLOT_MAX_RANGE = 900
PLOT_MAX_WIDTH = 300

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_title("LIDAR Sensor Mapping")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_xlim(-PLOT_MAX_WIDTH, PLOT_MAX_WIDTH)
ax.set_ylim(0, PLOT_MAX_RANGE)

try:
    while True:
        # Read response from Arduino
        line = arduino.readline().decode('utf-8').strip()
        if line and "Angle" in line:
            print(f"Received data: {line}")
            try:
                # Parse angle and distance
                angle_str, distance_str = line.split(", Distance: ")
                angle = int(angle_str.split(": ")[1].replace("°", ""))
                distance = distance_str.split(" ")[0]

                if distance == "ERROR":
                    print(f"Skipping invalid reading at angle {angle}")
                    continue

                # Angle offset 90
                angle = angle + 90
                print("Translated to trig coordinates with angle at", angle)
                
                distance = float(distance)

                if distance < MIN_RANGE_PLOT:
                    print(f"Skipping distance {distance} at angle {angle}")
                    continue

                x = distance * math.cos(math.radians(angle))
                if FLIP_X:
                    x = -x
                y = distance * math.sin(math.radians(angle))
                if y < 1 and y > -1:
                    y = 0
                    print("Flattened Y")
                if x < 1 and x > -1:
                    x = 0
                    print("Flattened X")
                print(f"Calculated coordinates -> X: {x}, Y: {y}, Angle: {angle}")

                # Update the dictionary of points
                points[angle] = (x, y)
                if DO_NOT_PLOT_ZEROS and distance == 0:
                    continue
                sc.set_offsets(list(points.values()))
                plt.pause(0.00001)
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