import cv2
import numpy as np

# Load calibration data
data = np.load('calibration_data.npz')
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

def undistort_image(frame):
    """
    Undistort the given frame using the loaded calibration data.

    Parameters:
    frame (numpy.ndarray): The input image frame to be undistorted.

    Returns:
    numpy.ndarray: The undistorted image frame.
    """
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    return undistorted_frame

cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = undistort_image(frame)

    # Concatenate the distorted and undistorted frames horizontally
    combined_frame = np.hstack((frame, undistorted_frame))

    cv2.imshow('Distorted and Undistorted Frames', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()