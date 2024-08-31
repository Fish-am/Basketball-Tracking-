import cv2
import numpy as np
import requests
from ultralytics import YOLOv10

# Initialize YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10x')

# Constants
HOOP_HEIGHT = 3.05  # Standard hoop height in meters
HOOP_WIDTH = 1.2    # Standard hoop width in meters
FRAME_HEIGHT = 480   # Height of the camera frame
FRAME_WIDTH = 640    # Width of the camera frame

# Camera calibration parameters (if available)
mtx = None  # Camera matrix
dist = None  # Distortion coefficients

# Function to capture frames from the ESP32 camera
def capture_frame():
    url = 'http://<ESP32_IP_ADDRESS>/stream'  # Replace with your ESP32's IP address
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Undistort the frame if calibration data is available
        if mtx is not None and dist is not None:
            frame = cv2.undistort(frame, mtx, dist)

        return frame
    return None

# Function to calculate trajectory based on ball positions
def calculate_trajectory(ball_positions):
    if len(ball_positions) < 2:
        return None  # Not enough data to calculate trajectory

    # Fit a second-degree polynomial (quadratic) to the ball positions
    x_coords = [pos[0] for pos in ball_positions]
    y_coords = [pos[1] for pos in ball_positions]
    coefficients = np.polyfit(x_coords, y_coords, 2)
    return coefficients  # Return the polynomial coefficients

# Function to analyze the trajectory and provide feedback
def analyze_trajectory(trajectory, hoop_position):
    a, b, c = trajectory

    # Calculate the vertex of the parabola (highest point)
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c

    # Check if the vertex height is sufficient to clear the hoop
    if vertex_y < HOOP_HEIGHT:
        return "Increase launch angle to clear the hoop."
    
    # Calculate where the trajectory intersects the hoop height
    intersection_x = (HOOP_HEIGHT - c) / a  # Solve for x when y = HOOP_HEIGHT
    if intersection_x < 0 or intersection_x > FRAME_WIDTH:
        return "Adjust your aim; the ball trajectory is too low or too high."

    # Determine horizontal position relative to hoop center
    hoop_center_x = (hoop_position[0] + hoop_position[2]) / 2  # Center of the hoop
    if intersection_x < hoop_center_x - (HOOP_WIDTH / 2):
        return "Aim more to the right."
    elif intersection_x > hoop_center_x + (HOOP_WIDTH / 2):
        return "Aim more to the left."
    
    return "Good shot trajectory!"

# Main loop
ball_positions = []  # List to store ball positions
hoop_position = None  # Store the hoop position

while True:
    # Capture frame from the ESP32 camera
    frame = capture_frame()
    if frame is None:
        continue

    # Process frame using YOLOv10
    results = model(frame)

    # Extract ball and hoop positions
    ball_positions.clear()  # Clear previous ball positions for the new shot
    for result in results:
        for detection in result.boxes:
            if detection.cls == 'basketball':
                ball_position = (detection.xyxy[0], detection.xyxy[1])  # Get (x, y)
                ball_positions.append(ball_position)
            elif detection.cls == 'hoop':
                hoop_position = detection.xyxy  # Get hoop position

    # Calculate trajectory if we have enough positions and hoop detected
    if len(ball_positions) >= 2 and hoop_position is not None:
        trajectory = calculate_trajectory(ball_positions)

        # Analyze trajectory and generate feedback
        if trajectory is not None:
            feedback = analyze_trajectory(trajectory, hoop_position)
            print(feedback)  # Display feedback (could also send to a server)

    # Display the video feed with annotations
    cv2.imshow('Basketball Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
