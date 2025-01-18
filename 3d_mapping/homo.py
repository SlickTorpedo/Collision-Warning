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

def click_event(event, x, y, flags, param):
    """
    Mouse callback function to log the coordinates of points clicked on the image.

    Parameters:
    event: The type of mouse event.
    x: The x-coordinate of the mouse event.
    y: The y-coordinate of the mouse event.
    flags: Any relevant flags passed by OpenCV.
    param: Any extra parameters supplied by OpenCV.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

img = cv2.imread("calibration_images/image_0.jpg")
if img is None:
    raise FileNotFoundError("Image file not found or failed to load.")

undistorted_img = undistort_image(img)

cv2.imshow("Undistorted Image", undistorted_img)
cv2.setMouseCallback("Undistorted Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()