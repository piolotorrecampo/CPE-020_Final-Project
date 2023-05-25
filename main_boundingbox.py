import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set camera parameters for low light conditions
cap = cv2.VideoCapture('C:/Users/Papzi/Downloads/vid1_edsa.mp4')
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # disable auto exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -8)  # set exposure to a lower value

# Set parameters for distance estimation
KNOWN_WIDTH = 3.5  # known width of the vehicle in meters
FOCAL_LENGTH = 800  # focal length of the camera in pixels

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply gamma correction to enhance image visibility
    gamma = 2.0
    frame = np.power(frame / 255.0, gamma)
    frame = (frame * 255).astype(np.uint8)

    # Convert BGR image to RGB and run YOLOv5 detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, size=640)

    # Process detection results
    for result in results.xyxy[0]:
        label_index = int(result[5])
        label = model.model.names[label_index]
        confidence = result[4]
        if label == 'car' and confidence > 0.5:  # only show cars with confidence > 0.5
            xmin, ymin, xmax, ymax = result[:4].int()
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Estimate distance using width of bounding box and known width of the vehicle
            vehicle_width = xmax - xmin
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / vehicle_width
            distance_str = f'{distance:.2f}m'

            # Display label and distance in bounding box
            label_str = f'{label} ({distance_str})'
            cv2.putText(frame, label_str, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output frame
    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
