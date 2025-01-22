import cv2
import json
import numpy as np
import math
import serial
import onnxruntime
import pandas as pd
from matplotlib import pyplot as plt

# --- The adjustable stuff ---
MAX_CAP_DISTANCE = 100
FLIP_FRAME = True
USE_WEBCAM = True
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256
LIDAR_TRACKING = True
# Don't touch anything after this point unless you know what you're doing

MODEL_FILEPATH = "./jetpack.onnx"

#The lidar is controlled by a servo and an arduino.
#We move the servo, take a measurement, and then store the distance in the LIDAR_TRACKED_OBJECTS dictionary.
#All communication is done between the arduino and the raspberry pi using serial communication.

#This is all the ID's that you use for the X axis on the graph of all the points.
#JetPack will output Y coordinates at these X coordinates and you can use this to plot them.
X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

def parse_image(frame):
    """
    This function takes in a frame and parses it into 6 different images.
    The model requires a few different images as input so we need to split the frame into these images.
    """
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
    """This function takes in a dataframe and seperates the points and standard deviation values into two dataframes."""
    points = df.iloc[lambda x: x.index % 2 == 0]
    std = df.iloc[lambda x: x.index % 2 != 0]
    points = pd.concat([points], ignore_index = True)
    std = pd.concat([std], ignore_index = True)

    return points, std

def extract_lead_car_info(lead_car, lead_car_prob):
    """This function is deprecated and will be removedi n the future.

    The lead car kinda works and more or less lets you know when it's leaving but the distance estimation is really off.
    Even with an offset it still gives a bad measurement. Since we are switching to LiDAR, this isn't needed anymore."""
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

def find_arduino():
    #print("Connecting to Arduino on /dev/cu.usbserial-2110...")
    print("Connecting to Arduino on /dev/cu.usbserial-110...")
    try:
        #ser = serial.Serial('/dev/cu.usbserial-2110', 115200, timeout=1)
        ser = serial.Serial('/dev/cu.usbserial-110', 115200, timeout=1)
        return ser
    except Exception as e:
        raise Exception(f"Failed to connect to Arduino: {e}")
    
def coordinates_to_angle(x, y):
    if isinstance(x, pd.Series) and len(x) == 1:
        x = float(x.iloc[0])
    if isinstance(y, pd.Series) and len(y) == 1:
        y = float(y.iloc[0])
    angle = math.degrees(math.atan2(y, x))
    return angle

def angle_to_coordinates(angle, distance):
    x = distance * math.cos(math.radians(angle))
    y = distance * math.sin(math.radians(angle))
    return x, y

def lidar_scan(arduino, x, y):
    if LIDAR_TRACKING:
        # print("X:", x, "Y:", y)
        angle = coordinates_to_angle(x, y)
        print("Extracted angle:", angle)
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
                    return -1

                distance = float(distance)

                return distance
            except ValueError as ve:
                print(f"Error parsing data: {line} -> {ve}")
                return -1
        else:
            print("No valid data received from Arduino.")
            return -1
    else:
        return -1

def main():

    try:
        arduino = find_arduino()
    except Exception as e:
        print(f"Error: {e}")
        exit()
        
    dummy_starter_data_sent = False #For some reason you have to send something because the first one always fails

    model = MODEL_FILEPATH
    
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/curvy.mov')

    parsed_images = []

    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    dim = (width, height)
    
    #These are all the spots where the data comes out, then you can use the X_IDXS to plot them.
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


    #Start a new session. Since it's an RNNT model, we need to keep the session open.
    session = onnxruntime.InferenceSession(model, None)
    
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()

    frame_counter = 0
    #Origionally this was used to track the frames so we could parse every other frame to save memory.
    #But jetpack is light enough now that we don't really have to worry about that so now it just kinda hangs out.

    while(cap.isOpened()):

        ret, frame = cap.read()
        if FLIP_FRAME:
            frame = cv2.flip(frame, -1)
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
                ax.clear()  # Clear the previous plot

                ax.scatter(middle_rotated, X_IDXS, color = "g")
                ax.scatter(points_ll_t2_rotated, X_IDXS, color = "y")
                ax.scatter(points_l_t_rotated, X_IDXS, color = "y")

                if LIDAR_TRACKING:
                    last_angle = 0
                    min_angle_change = 30
                    FILTER_90S = True
                    for x in X_IDXS:
                        if x > MAX_CAP_DISTANCE:
                            break
                        y = middle_rotated[X_IDXS == x]
                        y = y[0] if len(y) > 0 else 0

                        angle = coordinates_to_angle(x, y)

                        if FILTER_90S and (angle == 90 or angle == -90):
                            continue

                        if abs(angle - last_angle) < min_angle_change:
                            continue
                        last_angle = angle

                        distance = lidar_scan(arduino, x, y)
                        print("Distance:", distance, "cm")
                        #From CM to meters
                        distance /= 100

                        #Plot the points
                        coordinates_plottable = angle_to_coordinates(angle, distance)
                        ax.scatter(coordinates_plottable[1], coordinates_plottable[0], color = "b")
                        
                        

                ax.set_title("Road lines")
                ax.set_xlabel("red - road lines | green - predicted path | yellow - lane lines | blue - LiDAR")
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

            frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()