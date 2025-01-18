import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd
from ultralytics import YOLO


# --- JETPACK ---
MAX_CAP_DISTANCE = 200
FLIP_FRAME = True
USE_WEBCAM = True
CORRECT_FRAME = True
FLIP_COORDINATES = False
SCALE_COORDINATES = False
DOT_SCALE_AMOUNT = 1

#--- YOLO ---
IMG_SIZE = 640
CONFIDENCE_MIN = 0.4
PERFORM_YOLO = False

# --- JETPLANE ---
KNOWN_WIDTH = 1.5
FOCAL_LENGTH = 800
PERFORM_JETPLANE = True
PERFORM_INTERSECTION_CHECK = True

# --- IMAGE PARAMS ---
CROP_IMAGE = False
CROP_AMOUNT = 275

X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

# Real-world (ground plane) coordinates (float32)
# Meter by meter square
squareCorners2D = np.array([
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1]
], dtype=np.float32)

imageCorners2D = np.array([
    [773, 719],
    [757, 759],
    [1007, 753],
    [985, 715]
], dtype=np.float32)

# Load calibration data
data = np.load('3d_mapping/calibration_data.npz')
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

yolo_model = YOLO("yolov8n.pt")

def process_frame_yolo(frame):
    results = yolo_model(frame, verbose=False, imgsz=IMG_SIZE, conf=CONFIDENCE_MIN)
    
    object_locations = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy
            for coords in xyxy:
                x1, y1, x2, y2 = coords
                c = yolo_model.names[int(box.cls)]
                conf = box.conf
                object_locations.append((int(x1), int(y1), int(x2), int(y2), c, conf.item()))
                
    return object_locations

# Function to calculate distance from the camera to the vehicle
def calculate_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

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

def calculateHomographyMatrix(squareCorners2D, imageCorners2D):
    H, _ = cv2.findHomography(squareCorners2D, imageCorners2D, method=cv2.RANSAC)
    return H

def translatePoints(H, newPointsGround):
    # cv2.perspectiveTransform expects shape = (1, N, 2)
    newPointsGround_reshaped = newPointsGround[np.newaxis, :, :]

    # Project these points into the image using the homography
    newPointsInImage = cv2.perspectiveTransform(newPointsGround_reshaped, H)

    # newPointsInImage now has shape (1, N, 2)
    newPointsInImage = newPointsInImage[0]  # shape = (N, 2)

    return newPointsInImage

def parse_image(frame):
    H = (frame.shape[0]*2)//3
    W = frame.shape[1]
    parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
    parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

    return parsed

def seperate_points_and_std_values(df):
    points = df.iloc[lambda x: x.index % 2 == 0]
    std = df.iloc[lambda x: x.index % 2 != 0]
    points = pd.concat([points], ignore_index = True)
    std = pd.concat([std], ignore_index = True)

    return points, std

def extract_lead_car_info(lead_car, lead_car_prob):
    # Extract lead car information
    lead_car_info = []
    for i in range(2):  # 2 hypotheses
        hypothesis = {}
        start_idx = i * 51
        hypothesis['y_position'] = lead_car[start_idx:start_idx + 12:2].values
        hypothesis['x_position'] = lead_car[start_idx + 1:start_idx + 12:2].values
        hypothesis['speed'] = lead_car[start_idx + 12:start_idx + 24:2].values
        hypothesis['acceleration'] = lead_car[start_idx + 13:start_idx + 24:2].values
        hypothesis['x_position_std'] = lead_car[start_idx + 24:start_idx + 36:2].values
        hypothesis['y_position_std'] = lead_car[start_idx + 25:start_idx + 36:2].values
        hypothesis['speed_std'] = lead_car[start_idx + 36:start_idx + 48:2].values
        hypothesis['acceleration_std'] = lead_car[start_idx + 37:start_idx + 48:2].values
        lead_car_info.append(hypothesis)

    # Extract lead car probabilities
    lead_car_probabilities = lead_car_prob.values

    return lead_car_info, lead_car_probabilities

def main():
    model = "jetpack.onnx"
    
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/leading_car.mov')
    parsed_images = []

    width = 512
    height = 256
    dim = (width, height)
    
    plan_start_idx = 0
    plan_end_idx = 4955
    
    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528
    
    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8
    
    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264

    lead_start_idx = road_end_idx
    lead_end_idx = lead_start_idx + 55

    lead_prob_start_idx = lead_end_idx
    lead_prob_end_idx = lead_prob_start_idx + 3

    session = onnxruntime.InferenceSession(model, None)
    
    # plt.ion()  # Enable interactive mode
    # fig, ax = plt.subplots()

    frame_counter = 0

    homography_matrix = calculateHomographyMatrix(squareCorners2D, imageCorners2D)

    while(cap.isOpened()):

        ret, frame = cap.read()
        if CORRECT_FRAME:
            frame = undistort_image(frame)
        if FLIP_FRAME:
            frame = cv2.flip(frame, -1)
        if CROP_IMAGE:
            #Crop from the bottom CROP_AMOUNT
            frame = frame[:-275, :]
        if (ret == False):
            break

        if frame is not None:
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            parsed = parse_image(img_yuv)
    
        if (len(parsed_images) >= 2):
            del parsed_images[0]
    
        parsed_images.append(parsed)

        if (len(parsed_images) >= 2):
        
            parsed_arr = np.array(parsed_images)
            parsed_arr.resize((1,12,128,256))

            data = json.dumps({'data': parsed_arr.tolist()})
            data = np.array(json.loads(data)['data']).astype('float32')
            
            input_imgs = session.get_inputs()[0].name
            big_input_imgs = session.get_inputs()[1].name
            desire = session.get_inputs()[2].name
            traffic_convention = session.get_inputs()[3].name
            initial_state = session.get_inputs()[4].name
            output_name = session.get_outputs()[0].name
            
            big_input_imgs_data = np.zeros((1, 12, 128, 256)).astype('float32')
            
            desire_data = np.array([0]).astype('float32')
            desire_data.resize((1,8))
            
            traffic_convention_data = np.array([0, 0]).astype('float32')
            traffic_convention_data.resize((1,2))
            
            initial_state_data = np.zeros((1, 512)).astype('float32')  # Adjust the size as needed

            result = session.run([output_name], {input_imgs: data,
                                                big_input_imgs: big_input_imgs_data,
                                                desire: desire_data,
                                                traffic_convention: traffic_convention_data,
                                                initial_state: initial_state_data
                                                })

            res = np.array(result)

            lanes = res[:,:,lanes_start_idx:lanes_end_idx]
            lane_road = res[:,:,road_start_idx:road_end_idx]

            lanes_flat = lanes.flatten()
            df_lanes = pd.DataFrame(lanes_flat)

            ll_t = df_lanes[0:66]
            ll_t2 = df_lanes[66:132]
            points_ll_t, std_ll_t = seperate_points_and_std_values(ll_t)
            points_ll_t2, std_ll_t2 = seperate_points_and_std_values(ll_t2)

            l_t = df_lanes[132:198]
            l_t2 = df_lanes[198:264]
            points_l_t, std_l_t = seperate_points_and_std_values(l_t)
            points_l_t2, std_l_t2 = seperate_points_and_std_values(l_t2)

            r_t = df_lanes[264:330]
            r_t2 = df_lanes[330:396]
            points_r_t, std_r_t = seperate_points_and_std_values(r_t)
            points_r_t2, std_r_t2 = seperate_points_and_std_values(r_t2)

            rr_t = df_lanes[396:462]
            rr_t2 = df_lanes[462:528]
            points_rr_t, std_rr_t = seperate_points_and_std_values(rr_t)
            points_rr_t2, std_rr_t2 = seperate_points_and_std_values(rr_t2)

            road_flat = lane_road.flatten()
            df_road = pd.DataFrame(road_flat)

            roadr_t = df_road[0:66]
            roadr_t2 = df_road[66:132]
            points_road_t, std_ll_t = seperate_points_and_std_values(roadr_t)
            points_road_t2, std_ll_t2 = seperate_points_and_std_values(roadr_t2)

            roadl_t = df_road[132:198]
            roadl_t2 = df_road[198:264]
            points_roadl_t, std_rl_t = seperate_points_and_std_values(roadl_t)
            points_roadl_t2, std_rl_t2 = seperate_points_and_std_values(roadl_t2)

            lead_car_flat = res[:,:,lead_start_idx:lead_end_idx].flatten()
            lead_car = pd.DataFrame(lead_car_flat)

            lead_car_prob_flat = res[:,:,lead_prob_start_idx:lead_prob_end_idx].flatten()
            lead_car_prob = pd.DataFrame(lead_car_prob_flat)

            lead_car_info, lead_car_probabilities = extract_lead_car_info(lead_car, lead_car_prob)

            # print("Lead Car Information:")
            # for i, hypothesis in enumerate(lead_car_info):
            #     print(f"Hypothesis {i + 1}:")
            #     print("X Position:", hypothesis['x_position'])
            #     print("Y Position:", hypothesis['y_position'])
            #     print("Speed:", hypothesis['speed'])
            #     print("Acceleration:", hypothesis['acceleration'])
            #     print("X Position Std:", hypothesis['x_position_std'])
            #     print("Y Position Std:", hypothesis['y_position_std'])
            #     print("Speed Std:", hypothesis['speed_std'])
            #     print("Acceleration Std:", hypothesis['acceleration_std'])

            # print("Lead Car Probabilities:", lead_car_probabilities)

            middle = points_ll_t2.add(points_l_t, fill_value=0) / 2

            # Rotate points by 180 degrees
            middle_rotated = -middle
            points_ll_t2_rotated = -points_ll_t2
            points_l_t_rotated = -points_l_t

            if True:
            # if frame_counter % 2 == 0:
                # ax.clear()  # Clear the previous plot

                # ax.scatter(middle_rotated, X_IDXS, color = "g")
                # # print("MiddleX:", middle_rotated)
                # # print("MiddleY:", X_IDXS)
                # ax.scatter(points_ll_t2_rotated, X_IDXS, color = "y")
                # ax.scatter(points_l_t_rotated, X_IDXS, color = "y")

                X_IDXS_copy = X_IDXS.copy()
                #Flip all of them over the y-axis
                if FLIP_COORDINATES:
                    X_IDXS_copy = X_IDXS_copy * -1
                if SCALE_COORDINATES:
                    X_IDXS_copy = X_IDXS_copy * DOT_SCALE_AMOUNT

                points_road_t_rotated_matrix = np.array(points_road_t)
                points_road_t_rotated_matrix = np.column_stack((points_road_t_rotated_matrix, X_IDXS_copy))

                # points_roadl_t2_rotated_matrix = np.array(points_roadl_t2)
                # points_roadl_t2_rotated_matrix = np.column_stack((points_roadl_t2_rotated_matrix, X_IDXS_copy))

                points_middle_rotated_matrix = np.array(middle_rotated)
                points_middle_rotated_matrix = np.column_stack((points_middle_rotated_matrix, X_IDXS_copy))

                points_ll_t2_rotated_matrix = np.array(points_ll_t2_rotated)
                points_ll_t2_rotated_matrix = np.column_stack((points_ll_t2_rotated_matrix, X_IDXS_copy))

                points_l_t_rotated_matrix = np.array(points_l_t_rotated)
                points_l_t_rotated_matrix = np.column_stack((points_l_t_rotated_matrix, X_IDXS_copy))
                
                points_road_t_rotated_translated = translatePoints(homography_matrix, points_road_t_rotated_matrix)
                # points_roadl_t2_rotated_matrix_translated = translatePoints(homography_matrix, points_roadl_t2_rotated_matrix)
                points_middle_rotated_translated = translatePoints(homography_matrix, points_middle_rotated_matrix)
                points_ll_t2_rotated_translated = translatePoints(homography_matrix, points_ll_t2_rotated_matrix)
                points_l_t_rotated_matrix_translated = translatePoints(homography_matrix, points_l_t_rotated_matrix)

                green_dot_positions = []

                for idx, point in enumerate(points_road_t_rotated_translated):
                    x, y = map(int, point)
                    if idx <= MAX_CAP_DISTANCE:
                        cv2.circle(frame, (x, y), radius=5, color=(255, 255, 0), thickness=-1)
                    else:
                        cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

                # for idx, point in enumerate(points_roadl_t2_rotated_matrix_translated):
                #     x, y = map(int, point)
                #     if idx <= MAX_CAP_DISTANCE:
                #         cv2.circle(frame, (x, y), radius=5, color=(255, 255, 0), thickness=-1)
                #     else:
                #         cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

                for idx, point in enumerate(points_middle_rotated_translated):
                    x, y = map(int, point)
                    if idx <= MAX_CAP_DISTANCE:
                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                        green_dot_positions.append((x, y))
                    else:
                        cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

                for idx, point in enumerate(points_ll_t2_rotated_translated):
                    x, y = map(int, point)
                    if idx <= MAX_CAP_DISTANCE:
                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                        green_dot_positions.append((x, y))
                    else:
                        cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

                for idx, point in enumerate(points_l_t_rotated_matrix_translated):
                    x, y = map(int, point)
                    if idx <= MAX_CAP_DISTANCE:
                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                        green_dot_positions.append((x, y))
                    else:
                        cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

                # ax.set_title("Road lines")
                # ax.set_xlabel("red - road lines | green - predicted path | yellow - lane lines | pink - relevant car")
                # ax.set_ylabel("Range")
                #Flip the y-axis
                # ax.set_ylim(-5, MAX_CAP_DISTANCE)  # Set vertical axis limits
                # ax.set_xlim(-20, 20)  # Set horizontal axis limits
                #Mirror the x-axis
                # ax.invert_xaxis()

                # Convert frame to RGB and display it using Matplotlib
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #ax.imshow(frame_rgb, aspect='auto', extent=[-6, 8, 0, height])

                # plt.draw()
                # plt.pause(0.0001)

            if PERFORM_YOLO:
                object_locations = process_frame_yolo(frame)
            else:
                object_locations = []

            for loc in object_locations:
                x1, y1, x2, y2, c, conf = loc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, c, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                if PERFORM_JETPLANE:
                    estimate_distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, x2 - x1) #Use JetPlane to estimate the distance
                    cv2.putText(frame, f"Distance: {estimate_distance:.2f}m", (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                if PERFORM_INTERSECTION_CHECK:
                    for dot in green_dot_positions:
                        if x1 < dot[0] < x2 and y1 < dot[1] < y2:
                            cv2.putText(frame, "Intersection", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                            break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()