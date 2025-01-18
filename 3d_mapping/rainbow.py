import cv2
import numpy as np
import glob

# Dimensions of the chessboard (number of internal corners)
grid_size = (9, 6)

# The real-world size of each square (2 cm)
square_size = 2  # cm

# 3D world coordinates of the chessboard
obj_points = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

# Lists for storing the necessary points for calibration
object_points = []  # 3D world coordinates
image_points = []  # 2D image coordinates

# Folder where calibration images are stored
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

    if ret:
        object_points.append(obj_points)
        image_points.append(corners)

        # Visualize the corners
        cv2.drawChessboardCorners(img, grid_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Print the calibration results
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)