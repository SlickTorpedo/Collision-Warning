import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

IMG_SIZE = 640
CONFIDENCE_MIN = 0.4

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def process_frame_yolo(frame):
    results = model(frame, verbose=False, imgsz=IMG_SIZE, conf=CONFIDENCE_MIN)
    
    object_locations = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy
            for coords in xyxy:
                x1, y1, x2, y2 = coords
                c = model.names[int(box.cls)]
                conf = box.conf
                object_locations.append((int(x1), int(y1), int(x2), int(y2), c, conf.item()))
                
    return object_locations

# Constants (example values; replace with actual measurements)
KNOWN_WIDTH = 2.0  # Known width of the vehicle in meters
FOCAL_LENGTH = 800  # Focal length of the camera in pixels
known_width = 2.0
pixel_width = 100

# Function to calculate distance from the camera to the vehicle
def calculate_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

# Function to calculate lateral position
def calculate_lateral_position(image_center_x, bounding_box_center_x, distance):
    # Calculate the difference in pixels between image center and bounding box center
    delta_x_pixels = bounding_box_center_x - image_center_x
    # Convert pixel difference to meters (assuming square pixels)
    delta_x_meters = (delta_x_pixels * known_width) / pixel_width
    return delta_x_meters

stream = cv2.VideoCapture("daytime_footage.mov")

# Initialize the plot
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [], c='red', label='Vehicle Position')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.set_title('Overhead View of Vehicle Position')
ax.set_xlabel('Lateral Position (meters)')
ax.set_ylabel('Distance from Camera (meters)')
ax.invert_yaxis()  # Invert y-axis to have the camera at the bottom
ax.legend()

# Set fixed limits for the axes
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

while stream.isOpened():
    ret, img = stream.read()
    if not ret:
        break

    # Crop the bottom 300 pixels off the frame
    img = img[:-275, :]

    # Load image and get its dimensions
    image_height, image_width = img.shape[:2]
    image_center_x = image_width / 2

    object_locations = process_frame_yolo(img)

    # Clear previous data
    lateral_positions = []
    distances = []

    for loc in object_locations:
        x1, y1, x2, y2, c, conf = loc
        if c == "car":
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            bounding_box = {'left': x1, 'top': y1, 'right': x2, 'bottom': y2}
            bounding_box_width = bounding_box['right'] - bounding_box['left']
            bounding_box_center_x = (bounding_box['left'] + bounding_box['right']) / 2

            # Calculate distance and lateral position
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, bounding_box_width)
            lateral_position = calculate_lateral_position(image_center_x, bounding_box_center_x, distance)

            # Append the new data
            lateral_positions.append(lateral_position)
            distances.append(distance)

    # Update the plot data
    sc.set_offsets(np.c_[lateral_positions, distances])
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Display the image
    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()