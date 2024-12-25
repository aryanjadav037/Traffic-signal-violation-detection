import os
import cv2
import torch
import numpy as np
import time
import csv  # Add this import for CSV functionality
from ultralytics import YOLO  # YOLOv8 model import

# Load YOLOv8 model (change to 'yolov8s.pt' for better accuracy if needed)
model = YOLO('yolov8n.pt')

# Video feed
cap = cv2.VideoCapture("1.mp4")

# Get FPS and set desired frame size
fps = cap.get(cv2.CAP_PROP_FPS)
DESIRED_WIDTH, DESIRED_HEIGHT = 1280, 720

# Regions and thresholds
TRAFFIC_LIGHT_REGION = (400, 150, 47, 150)
RED_THRESHOLD = 200
STOP_LINE_Y = 500  # Position of the stop line

# Tracking state and counters
vehicle_states = {}  # Store vehicle state by ID
violated_vehicles = set()  # Store IDs of vehicles that violated
ID_COUNTER = 0  # Counter for unique vehicle IDs
violation_count = 0  # Total violation counter

# Define curve parameters
CURVE_START_X = 800  # Start of the curve on x-axis
CURVE_END_X = 1280  # End of the curve on x-axis
CURVE_HEIGHT = 50  # Height of the curve

# Directory for saving violation images
VIOLATION_DIR = "violations_images"
os.makedirs(VIOLATION_DIR, exist_ok=True)

# Open CSV file and create a writer
CSV_FILE_PATH = "violations_log.csv"
csv_file = open(CSV_FILE_PATH, mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Write the header row to the CSV file
csv_writer.writerow(["Vehicle ID", "Timestamp", "Centroid", "Violation Type"])

def is_red_light(frame):
    """Detect if the traffic light is red based on color intensity."""
    x, y, w, h = TRAFFIC_LIGHT_REGION
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])

    # Combine masks to detect red pixels
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_pixels = cv2.countNonZero(mask1 + mask2)
    return red_pixels > RED_THRESHOLD

def detect_vehicles(frame):
    """Detect vehicles using YOLOv8."""
    results = model(frame)[0]  # Get detection results
    vehicles = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, _, cls = map(int, result[:6])
        label = model.names[cls]
        if label in ['car', 'motorcycle', 'bus', 'truck']:
            vehicles.append((x1, y1, x2, y2))
    return vehicles

def get_centroid(vehicle):
    """Calculate the centroid of a vehicle."""
    x1, y1, x2, y2 = vehicle
    return (x1 + x2) // 2, (y1 + y2) // 2

def track_vehicle(centroid, threshold=50):
    """Track vehicles and assign a unique ID to new ones."""
    global ID_COUNTER

    for vehicle_id, state in vehicle_states.items():
        prev_centroid = state['centroid']
        if np.linalg.norm(np.array(prev_centroid) - np.array(centroid)) < threshold:
            state['centroid'] = centroid
            return vehicle_id

    # Assign a new ID to a new vehicle
    ID_COUNTER += 1
    vehicle_states[ID_COUNTER] = {'centroid': centroid, 'path': [centroid], 'violated': False}
    return ID_COUNTER

def draw_combined_stop_line(frame, start_x, stop_line_y, end_x):
    """Draws a stop line with a straight portion and a curved portion."""
    straight_end_x = end_x - 600
    cv2.line(frame, (start_x, stop_line_y), (straight_end_x, stop_line_y), (255, 0, 0), 2)

    # Draw the curve
    curve_points = []
    for x in range(straight_end_x, end_x, 50):
        y_offset = CURVE_HEIGHT * ((x - straight_end_x) / 200) ** 2  # Curve equation
        curve_points.append((x, stop_line_y + int(y_offset)))

    # Draw curve
    for i in range(len(curve_points) - 1):
        cv2.line(frame, curve_points[i], curve_points[i + 1], (255, 0, 0), 2)

    return [(x, stop_line_y) for x in range(start_x, end_x)]

def is_centroid_crossing_stop_line(centroid, stop_line_points):
    """Check if the vehicle centroid has crossed the stop line path."""
    for (x, y) in stop_line_points:
        if centroid[0] > x and centroid[1] > y:
            return True
    return False

def detect_violations(vehicles, red_light_on, stop_line_points):
    """Detect and count unique violations."""
    global violation_count
    violations = []

    if red_light_on:
        for vehicle in vehicles:
            centroid = get_centroid(vehicle)
            vehicle_id = track_vehicle(centroid)

            # Check if the vehicle crosses the stop line path and hasn't violated yet
            if (is_centroid_crossing_stop_line(centroid, stop_line_points) and
                not vehicle_states[vehicle_id]['violated'] and
                not is_in_curve_area(centroid)):  # Ensure vehicle is not in curve area

                vehicle_states[vehicle_id]['violated'] = True  # Mark as violated
                violated_vehicles.add(vehicle_id)
                violation_count += 1  # Increment violation count
                violations.append(vehicle)

                # Log the violation
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                csv_writer.writerow([vehicle_id, timestamp, centroid, "Red Light Violation"])

                # Highlight violating vehicle on the frame
                x1, y1, x2, y2 = vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Violation ID: {vehicle_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Save the frame
                frame_path = os.path.join(VIOLATION_DIR, f"violation_{vehicle_id}_{timestamp}.jpg")
                cv2.imwrite(frame_path, frame)

    return violations

def is_in_curve_area(centroid):
    """Check if the centroid is in the curved area of the stop line."""
    return CURVE_START_X < centroid[0] < CURVE_END_X

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
    start_time = time.time()

    red_light_on = is_red_light(frame)  # Check traffic light status
    vehicles = detect_vehicles(frame)  # Detect vehicles in the frame

    stop_line_points = draw_combined_stop_line(frame, 400, STOP_LINE_Y, frame.shape[1])  # Draw stop line and get points

    violations = detect_violations(vehicles, red_light_on, stop_line_points)  # Detect violations

    # Draw detections and violations
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle
        color = (0, 255, 0)  # Default color for non-violators

        vehicle_centroid = get_centroid(vehicle)
        vehicle_id = track_vehicle(vehicle_centroid)

        # Highlight violators in red
        if vehicle_id in violated_vehicles:
            color = (0, 0, 255)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display status and statistics
    status = "RED LIGHT" if red_light_on else "GREEN LIGHT"
   # Displaying the frame with annotations

    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Violations: {violation_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the frame
    cv2.imshow("Red Light Violation Detection", frame)

# Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close CSV file after processing
csv_file.close()
