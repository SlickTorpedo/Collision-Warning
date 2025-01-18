import cv2
import numpy as np

# Real-world (ground plane) coordinates (float32)
squareCorners2D = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
], dtype=np.float32)

# Corresponding pixel coordinates in the image (float32)
imageCorners2D = np.array([
    [727, 676],
    [937, 673],
    [946, 709],
    [702, 717]
], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(squareCorners2D, imageCorners2D, method=cv2.RANSAC)
# or H = cv2.getPerspectiveTransform(squareCorners2D, imageCorners2D)
print("Homography Matrix:")
print(H)

#Read the points from ../coordinates.csv
points = np.loadtxt('../coordinates.csv', delimiter=',', skiprows=1)
#Skip the first colum
newPointsGround = points[:, 1:]
# newPointsGround = np.array([
#     [0.5, 0.5],
#     [1.5, 0.5],
#     [2.5, 0.5],
# ], dtype=np.float32)

# cv2.perspectiveTransform expects shape = (1, N, 2)
newPointsGround_reshaped = newPointsGround[np.newaxis, :, :]

# Project these points into the image using the homography
newPointsInImage = cv2.perspectiveTransform(newPointsGround_reshaped, H)

# newPointsInImage now has shape (1, N, 2)
newPointsInImage = newPointsInImage[0]  # shape = (N, 2)

img = cv2.imread("calibration_images/image_0.jpg")
if img is None:
    raise FileNotFoundError("Image file not found or failed to load.")

# Draw the projected points on the image
for point in newPointsInImage:
    x, y = map(int, point)  # Convert to integers
    cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

# Display the result
cv2.imshow("Projected Points", img)
cv2.waitKey(0)
cv2.destroyAllWindows()