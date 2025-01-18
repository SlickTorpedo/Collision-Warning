import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO

# ----- TUNABLES -----

#JETPACK
MAX_CAP_DISTANCE = 100
FLIP_FRAME = False
USE_WEBCAM = False
CROP_FRAME = True #Remove most of the hood 
CROP_AMOUNT = 275

#YOLO
IMG_SIZE = 640
CONFIDENCE_MIN = 0.4

#MERCURY
KNOWN_WIDTH = 3  # Known width of the vehicle in meters
FOCAL_LENGTH = 1000  # Focal length of the camera in pixels
known_width = 3 # Known width of the vehicle in meters (maybe)
pixel_width = 100 # Pixel width of the vehicle in the image (maybe)

#----- END TUNABLES -----

vision_model = None

X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

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


def process_frame_yolo(frame):
    results = vision_model(frame, verbose=False, imgsz=IMG_SIZE, conf=CONFIDENCE_MIN)
    
    object_locations = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy
            for coords in xyxy:
                x1, y1, x2, y2 = coords
                c = vision_model.names[int(box.cls)]
                conf = box.conf
                object_locations.append((int(x1), int(y1), int(x2), int(y2), c, conf.item()))
                
    return object_locations

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

def main():
    global vision_model
    vision_model = YOLO("yolov8n.pt")

    model = "jetpack.onnx"
    
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/daytime_footage.mov')
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
    
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()

    frame_counter = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if FLIP_FRAME:
            frame = cv2.flip(frame, -1)
        if CROP_FRAME:
            frame = frame[:-CROP_AMOUNT, :]
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

            #YOLO STUFF ---
            image_height, image_width = img.shape[:2]
            image_center_x = image_width / 2

            object_locations = process_frame_yolo(frame)

            # Clear previous data
            lateral_positions = []
            distances = []

            for loc in object_locations:
                x1, y1, x2, y2, c, conf = loc
                if c: #You can change this to if c == "car" or whatever if you only want to look for cars.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, c, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    yolo_bounding_box = {'left': x1, 'top': y1, 'right': x2, 'bottom': y2}
                    yolo_bounding_box_width = yolo_bounding_box['right'] - yolo_bounding_box['left']
                    yolo_bounding_box_center_x = (yolo_bounding_box['left'] + yolo_bounding_box['right']) / 2

                    # Calculate distance and lateral position
                    distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, yolo_bounding_box_width)
                    lateral_position = calculate_lateral_position(image_center_x, yolo_bounding_box_center_x, distance)

                    # Append the new data
                    lateral_positions.append(lateral_position * -1) #Flip the x-axis
                    distances.append(distance)

            # END YOLO STUFF

            if True:
            # if frame_counter % 2 == 0:
                ax.clear()  # Clear the previous plot

                ax.scatter(middle_rotated, X_IDXS, color = "g")
                ax.scatter(points_ll_t2_rotated, X_IDXS, color = "y")
                ax.scatter(points_l_t_rotated, X_IDXS, color = "y")
                # ax.scatter(points_road_t, X_IDXS, color = "r")
                # ax.scatter(points_road_t2, X_IDXS, color = "r")

                #Plot the yolo data
                ax.scatter(lateral_positions, distances, color='red', label='Vehicle Position')

                # Plot lead car positions
                # for hypothesis in lead_car_info:
                #     ax.scatter(hypothesis['x_position'], hypothesis['y_position'], color='blue', label='Lead Car')
                #     print("Lead Car X:", hypothesis['x_position'])
                #     print("Lead Car Y:", hypothesis['y_position'])
                # print("X", lead_car_info[0]['x_position'])
                # print("Y", lead_car_info[0]['y_position'])
                for idx, car_x in enumerate(lead_car_info[0]['x_position']):
                    try:
                        # ax.scatter(car, lead_car_info[0]['y_position'][idx], color='blue', label='Lead Car')
                        # print("Percentage:", lead_car_probabilities[idx])
                        # ax.text(car, lead_car_info[0]['y_position'][idx], str(lead_car_probabilities[idx]))
                        if(lead_car_probabilities[idx] > 80):
                            ax.scatter(car_x, lead_car_info[0]['y_position'][idx], color='blue', label='Relevant Car')
                            ax.text(car_x, lead_car_info[0]['y_position'][idx], "Relevant Car")
                    except:
                        pass

                #ax.scatter(lead_car_info[0]['x_position'], lead_car_info[0]['y_position'], color='pink', label='Lead Car')
                # print("Percentages:", lead_car_probabilities)

                ax.set_title("Road lines")
                ax.set_xlabel("red - road lines | green - predicted path | yellow - lane lines | pink - relevant car")
                ax.set_ylabel("Range")
                #Flip the y-axis
                ax.set_ylim(-5, MAX_CAP_DISTANCE)  # Set vertical axis limits
                ax.set_xlim(-20, 20)  # Set horizontal axis limits
                #Mirror the x-axis
                ax.invert_xaxis()

                # Convert frame to RGB and display it using Matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #ax.imshow(frame_rgb, aspect='auto', extent=[-6, 8, 0, height])

                plt.draw()
                plt.pause(0.0001)

            cv2.imshow('frame', frame)
            cv2.imshow('img', img)

            frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()