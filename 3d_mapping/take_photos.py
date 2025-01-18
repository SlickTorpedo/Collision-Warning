# #Press space bar to take a photo and save it as calibration_images/image_{i}.jpg
# import cv2

# stream = cv2.VideoCapture(0)
# i = 0

# while True:
#     ret, frame = stream.read()
#     frame = cv2.flip(frame, -1)
#     cv2.imshow("Camera", frame)

#     key = cv2.waitKey(1)
#     if key == ord(' '):
#         cv2.imwrite(f"calibration_images/image_{i}.jpg", frame)
#         i += 1
#     elif key == 27:
#         break

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

stream = cv2.VideoCapture(0)
i = 0

while True:
    ret, frame = stream.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    undistorted_frame = undistort_image(frame)
    cv2.imshow("Camera", undistorted_frame)

    key = cv2.waitKey(1)
    if key == ord(' '):
        cv2.imwrite(f"calibration_images/image_{i}.jpg", undistorted_frame)
        i += 1
    elif key == 27:
        break

stream.release()
cv2.destroyAllWindows()