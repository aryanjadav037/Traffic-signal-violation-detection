
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO  # YOLOv8 model import

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Video feed
cap = cv2.VideoCapture("input.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get FPS and set desired frame size
fps = cap.get(cv2.CAP_PROP_FPS)
DESIRED_WIDTH, DESIRED_HEIGHT = 1280, 720

# Regions and thresholds
TRAFFIC_LIGHT_REGION = (130, 70, 170, 150)
RED_THRESHOLD = 200
STOP_LINE_Y = 500  # Position of the stop line

# Tracking state and counters
vehicle_states = {}  # Store vehicle state by ID
violated_vehicles = set()  # Store IDs of vehicles that violated
ID_COUNTER = 0  # Counter for unique vehicle IDs
violation_count = 0  # Total violation counter

# Store violation data for dashboard analysis
violation_data = []

# Define curve parameters
CURVE_START_X = 800  # Start of the curve on x-axis
CURVE_END_X = 1280  # End of the curve on x-axis
CURVE_HEIGHT = 50  # Height of the curve

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

def is_bounding_box_crossing_stop_line(vehicle, stop_line_y):
    """Check if the entire bounding box of the vehicle has crossed the stop line."""
    x1, y1, x2, y2 = vehicle
    return y2 > stop_line_y  # Check if the bottom of the bounding box is below the stop line

def detect_violations(vehicles, red_light_on, stop_line_points):
    """Detect and count unique violations."""
    global violation_count
    violations = []

    if red_light_on:
        for vehicle in vehicles:
            centroid = get_centroid(vehicle)
            vehicle_id = track_vehicle(centroid)

            # Check if the vehicle's bounding box crosses the stop line and hasn't violated yet
            if (is_bounding_box_crossing_stop_line(vehicle, STOP_LINE_Y) and
                not vehicle_states[vehicle_id]['violated'] and
                not is_in_curve_area(centroid)):  # Ensure vehicle is not in curve area

                vehicle_states[vehicle_id]['violated'] = True  # Mark as violated
                violated_vehicles.add(vehicle_id)
                violation_count += 1  # Increment violation count
                violations.append(vehicle)

                # Store the violation details
                violation_data.append({
                    'vehicle_id': vehicle_id,
                    'violation_time': time.time(),  # Store the time when the violation occurred
                })

    return violations

def is_in_curve_area(centroid):
    """Check if the centroid is in the curved area of the stop line."""
    return CURVE_START_X < centroid[0] < CURVE_END_X

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
    start_time = time.time()

    red_light_on = is_red_light(frame)  # Check traffic light status
    vehicles = detect_vehicles(frame)  # Detect vehicles in the frame

    stop_line_points = draw_combined_stop_line(frame, 200, STOP_LINE_Y, frame.shape[1])  # Draw stop line and get points

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
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if red_light_on else (0, 255, 0), 2)
    cv2.putText(frame, f"Violations: {violation_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    processing_time = time.time() - start_time
    cv2.putText(frame, f"Processing Time: {processing_time:.2f}s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Traffic Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save violation data for dashboard analysis
np.save('violation_data.npy', violation_data)

cap.release()
cv2.destroyAllWindows()
