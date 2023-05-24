import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MIDAS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Calling the video
cap = cv2.VideoCapture('C:/Users/Papzi/Downloads/vid1_edsa.mp4')

# Set camera parameters for low light conditions (this may vary depending to the sorrounding)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # disable auto exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -8)  # set exposure to a lower value

# Set parameters for distance estimation (di ko alam kung ano kunin yung values dito)
KNOWN_WIDTH = 3.5  # known width of the vehicle in meters
FOCAL_LENGTH = 800

# saving logs
logs = {'Number of Vehicles': [], 'State': [], 'Current Timeframe': []}
df = pd.DataFrame(logs)

# access distance variable outside of the loop
distance = None
depth_pred = None
current_time_str = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get current time in seconds (FOR DOCUMENTATION)
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    current_time_str = print(f"Current time: {current_time:.2f}s")

    # Apply gamma correction to enhance image visibility
    gamma = 2.0
    frame = np.power(frame / 255.0, gamma)
    frame = (frame * 255).astype(np.uint8)

    # Convert BGR image to RGB and run YOLOv5 detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, size=640)

    # Initialize vehicle count dictionary
    vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0}

    """
    # Create a named window for displaying depth map
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    """

    # Process detection results
    for result in results.xyxy[0]:
        label_index = int(result[5])
        label = model.model.names[label_index]
        confidence = result[4]

        if label in ['car', 'bus', 'truck'] and confidence > 0.5:  # only show cars with confidence > 0.5
            xmin, ymin, xmax, ymax = result[:4].int()
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Estimate distance using width of bounding box and known width of the vehicle
            vehicle_width = xmax - xmin

            # Estimate depth using MIDAS depth estimation model
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            input_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
            input_img = cv2.resize(input_img, (384, 384))
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).permute(0, 3, 1, 2).float()
            depth_pred = midas(input_tensor).cpu().squeeze().detach().numpy()
            depth_pred = cv2.resize(depth_pred, (int(bbox[2]), int(bbox[3])))
            depth_pred = np.clip(depth_pred, 0, 1000)

            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / vehicle_width * np.mean(depth_pred)
            distance_str = f'{distance:.2f}m'

            # Visualize depth prediction
            """depth_pred_norm = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
            depth_pred_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_pred_norm), cv2.COLORMAP_JET)
            depth_map = np.zeros_like(frame)
            depth_map[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :] = depth_pred_colormap
            depth_map = cv2.addWeighted(frame, 0.7, depth_map, 0.3, 0)
            cv2.imshow('Depth Map', depth_map)"""

            # Display label and distance in bounding box
            label_str = f'{label} ({distance_str})'
            cv2.putText(frame, label_str, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # print current time and distance of the vehicle
            print(current_time_str)
            print(label_str)

            # Increment vehicle count
            vehicle_counts[label] += 1

    current_vehicle_count = sum(vehicle_counts.values())
    vehicle_count_str = f'Total vehicles detected: {current_vehicle_count}'
    cv2.putText(frame, vehicle_count_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # CONDITION (the distance is not fixed)
    if current_vehicle_count == 1 and int(distance) < 16000:
        print(f"No. Vehicles: {current_vehicle_count} -> HEADLIGHT IS NOW IN DIM")
        logs['Number of Vehicles'].append(current_vehicle_count)
        logs['State'].append("HEADLIGHT IS NOW IN DIM")
    elif current_vehicle_count == 0:
        print(f"No. Vehicles: {current_vehicle_count} -> HEADLIGHT IS NOW IN FULL BRIGHT")
        logs['Number of Vehicles'].append(current_vehicle_count)
        logs['State'].append("HEADLIGHT IS NOW IN FULL BRIGHT")
    elif current_vehicle_count > 3:
        print(f"No. Vehicles: {current_vehicle_count} -> HEADLIGHT IS NOW IN DIM")
        logs['Number of Vehicles'].append(current_vehicle_count)
        logs['State'].append("HEADLIGHT IS NOW IN DIM")

    # Display output frame
    cv2.imshow('Vehicle Detection', frame)

    # close the windows the letter "q" is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# append the new row to the DataFrame
df = df._append(logs, ignore_index=True)

# save the updated DataFrame to the file
df.to_excel('logs.xlsx', index=False)

# Release resources
cap.release()
cv2.destroyAllWindows()
