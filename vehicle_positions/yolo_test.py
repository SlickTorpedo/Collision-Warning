from ultralytics import YOLO
import cv2

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

stream = cv2.VideoCapture("old.mp4")

while True:
    ret, img = stream.read()
    if not ret:
        break

    object_locations = process_frame_yolo(img)

    for loc in object_locations:
        x1, y1, x2, y2, c, conf = loc
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()