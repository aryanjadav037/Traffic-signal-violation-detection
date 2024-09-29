# import cv2
# import torch
# import numpy as np
# import pytesseract
# import os

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Video feed
# cap = cv2.VideoCapture('input_Number.mp4')

# # Traffic light region and red threshold
# TRAFFIC_LIGHT_REGION = (432, 100, 47, 150)
# RED_THRESHOLD = 200

# # Desired frame size
# DESIRED_WIDTH = 1280
# DESIRED_HEIGHT = 720

# # Update the path to where Tesseract is installed on your system
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Directory for saving violation images
# VIOLATION_DIR = 'violations'
# if not os.path.exists(VIOLATION_DIR):
#     os.makedirs(VIOLATION_DIR)

# # Set to keep track of vehicles already captured for violation
# saved_violations = set()

# def is_red_light(frame):
#     """Detect red light based on color intensity."""
#     x, y, w, h = TRAFFIC_LIGHT_REGION
#     roi = frame[y:y+h, x:x+w]
#     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
   
#     # Red color range in HSV
#     lower_red = np.array([0, 100, 100])
#     upper_red = np.array([10, 255, 255])
#     mask1 = cv2.inRange(hsv, lower_red, upper_red)

#     lower_red = np.array([160, 100, 100])
#     upper_red = np.array([180, 255, 255])
#     mask2 = cv2.inRange(hsv, lower_red, upper_red)
   
#     red_mask = mask1 + mask2
#     red_pixels = cv2.countNonZero(red_mask)
#     return red_pixels > RED_THRESHOLD

# def detect_vehicles(frame):
#     """Detect vehicles using YOLOv5."""
#     results = model(frame)
#     vehicles = []
   
#     for detection in results.xyxy[0]:
#         label = results.names[int(detection[5])]
#         if label in ['car', 'motorcycle', 'bus', 'truck']:
#             x1, y1, x2, y2 = map(int, detection[:4])
#             vehicles.append((x1, y1, x2, y2))
   
#     return vehicles

# def detect_violation(vehicles, red_light_on, stop_line_y):
#     """Detect vehicles crossing the stop line when the red light is on."""
#     violations = []
   
#     if red_light_on:
#         for (x1, y1, x2, y2) in vehicles:
#             if y2 > stop_line_y:
#                 violations.append((x1, y1, x2, y2))
   
#     return violations

# def recognize_license_plate(frame, vehicle):
#     """Recognize license plate using Tesseract OCR."""
#     x1, y1, x2, y2 = vehicle
#     vehicle_roi = frame[y1:y2, x1:x2]
   
#     # Preprocessing for OCR
#     gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     license_plate_text = pytesseract.image_to_string(thresh, config='--psm 8')
   
#     return license_plate_text.strip()

# def get_vehicle_id(vehicle):
#     """Generate a unique ID for a vehicle based on its coordinates."""
#     x1, y1, x2, y2 = vehicle
#     return f"{x1}-{y1}-{x2}-{y2}"

# def is_full_size_vehicle(vehicle, frame_shape, min_size_ratio=0.1):
#     """Check if the vehicle is full-size and large enough to capture."""
#     x1, y1, x2, y2 = vehicle
#     frame_height, frame_width = frame_shape[:2]
    
#     # Vehicle width and height
#     vehicle_width = x2 - x1
#     vehicle_height = y2 - y1
    
#     # Ensure the vehicle size is greater than a certain ratio of the frame (e.g., 10%)
#     if vehicle_width / frame_width < min_size_ratio or vehicle_height / frame_height < min_size_ratio:
#         return False
    
#     # Ensure the vehicle is fully within the frame
#     if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
#         return False
    
#     return True

# # Main loop
# STOP_LINE_Y = 500
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     # Resize the frame to the desired size
#     frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
   
#     red_light_on = is_red_light(frame)
#     vehicles = detect_vehicles(frame)
#     violations = detect_violation(vehicles, red_light_on, STOP_LINE_Y)
   
#     # Draw traffic light region
#     x, y, w, h = TRAFFIC_LIGHT_REGION
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
   
#     # Show traffic light status
#     status = "RED LIGHT" if red_light_on else "GREEN LIGHT"
#     cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if red_light_on else (0, 255, 0), 2)
   
#     # Draw vehicles and violations
#     for vehicle in vehicles:
#         x1, y1, x2, y2 = vehicle
#         color = (0, 255, 0)
#         if vehicle in violations:
#             color = (0, 0, 255)  # Red box for violations
            
#             # Generate a unique ID for the vehicle
#             vehicle_id = get_vehicle_id(vehicle)
            
#             # Save image of violation only if it hasn't been saved before and vehicle is full-sized
#             if vehicle_id not in saved_violations and is_full_size_vehicle(vehicle, frame.shape):
#                 saved_violations.add(vehicle_id)
#                 vehicle_roi = frame[y1:y2, x1:x2]
#                 violation_image_path = os.path.join(VIOLATION_DIR, f"violation_{frame_count}_{x1}_{y1}.jpg")
#                 cv2.imwrite(violation_image_path, vehicle_roi)
            
#             # Recognize license plate
#             license_plate = recognize_license_plate(frame, vehicle)
#             cv2.putText(frame, license_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
   
#     # Draw stop line
#     cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (255, 0, 0), 2)
   
#     # Display the result
#     cv2.imshow('Red Light Violation Detection', frame)
   
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
import numpy as np
import pytesseract
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video feed
cap = cv2.VideoCapture('Input.mp4')

# Traffic light region and red threshold
TRAFFIC_LIGHT_REGION = (432, 100, 47, 150)
RED_THRESHOLD = 200

# Desired frame size
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# Update the path to where Tesseract is installed on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory for saving violation images
VIOLATION_DIR = 'violations'
if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

# Set to keep track of vehicles already captured for violation
saved_violations = set()

def is_red_light(frame):
    """Detect red light based on color intensity."""
    x, y, w, h = TRAFFIC_LIGHT_REGION
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
   
    # Red color range in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
   
    red_mask = mask1 + mask2
    red_pixels = cv2.countNonZero(red_mask)
    return red_pixels > RED_THRESHOLD

def detect_vehicles(frame):
    """Detect vehicles using YOLOv5."""
    results = model(frame)
    vehicles = []
   
    for detection in results.xyxy[0]:
        label = results.names[int(detection[5])]
        if label in ['car', 'motorcycle', 'bus', 'truck']:
            x1, y1, x2, y2 = map(int, detection[:4])
            vehicles.append((x1, y1, x2, y2))
   
    return vehicles

def detect_violation(vehicles, red_light_on, stop_line_y):
    """Detect vehicles crossing the stop line when the red light is on."""
    violations = []
   
    if red_light_on:
        for (x1, y1, x2, y2) in vehicles:
            if y2 > stop_line_y:
                violations.append((x1, y1, x2, y2))
   
    return violations

def recognize_license_plate(frame, vehicle):
    """Recognize license plate using Tesseract OCR."""
    x1, y1, x2, y2 = vehicle
    vehicle_roi = frame[y1:y2, x1:x2]
   
    # Preprocessing for OCR
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    license_plate_text = pytesseract.image_to_string(thresh, config='--psm 8')
   
    return license_plate_text.strip()

def get_vehicle_id(vehicle):
    """Generate a unique ID for a vehicle based on its coordinates."""
    x1, y1, x2, y2 = vehicle
    return f"{x1}-{y1}-{x2}-{y2}"

def is_full_size_vehicle(vehicle, frame_shape, min_size_ratio=0.1):
    """Check if the vehicle is full-size and large enough to capture."""
    x1, y1, x2, y2 = vehicle
    frame_height, frame_width = frame_shape[:2]
    
    # Vehicle width and height
    vehicle_width = x2 - x1
    vehicle_height = y2 - y1
    
    # Ensure the vehicle size is greater than a certain ratio of the frame (e.g., 10%)
    if vehicle_width / frame_width < min_size_ratio or vehicle_height / frame_height < min_size_ratio:
        return False
    
    # Ensure the vehicle is fully within the frame
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return False
    
    return True

# Main loop
STOP_LINE_Y = 500
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize the frame to the desired size
    frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
   
    red_light_on = is_red_light(frame)
    vehicles = detect_vehicles(frame)
    violations = detect_violation(vehicles, red_light_on, STOP_LINE_Y)
   
    # Draw traffic light region
    x, y, w, h = TRAFFIC_LIGHT_REGION
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
   
    # Show traffic light status
    status = "RED LIGHT" if red_light_on else "GREEN LIGHT"
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if red_light_on else (0, 255, 0), 2)
   
    # Draw vehicles and violations
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle
        color = (0, 255, 0)
        if vehicle in violations:
            color = (0, 0, 255)  # Red box for violations
            
            # Generate a unique ID for the vehicle
            vehicle_id = get_vehicle_id(vehicle)
            
            # Save image of violation only if it hasn't been saved before and vehicle is full-sized
            if vehicle_id not in saved_violations and is_full_size_vehicle(vehicle, frame.shape):
                saved_violations.add(vehicle_id)
                vehicle_roi = frame[y1:y2, x1:x2]
                violation_image_path = os.path.join(VIOLATION_DIR, f"violation_{frame_count}_{x1}_{y1}.jpg")
                cv2.imwrite(violation_image_path, vehicle_roi)
            
            # Recognize license plate
            license_plate = recognize_license_plate(frame, vehicle)
            cv2.putText(frame, license_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
   
    # Draw stop line
    cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (255, 0, 0), 2)
   
    # Display the result
    cv2.imshow('Red Light Violation Detection', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
