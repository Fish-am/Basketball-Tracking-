import cv2
import numpy as np
import requests
from ultralytics import YOLOv10

model = YOLOv10('yolov10n.pt')

HOOP_HEIGHT = 3.05  
HOOP_WIDTH = 1.2    
FRAME_HEIGHT = 480  
FRAME_WIDTH = 640   

mtx = None  
dist = None 

def capture_frame():
    url = 'http://<ESP32_IP_ADDRESS>/stream' 
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if mtx is not None and dist is not None:
            frame = cv2.undistort(frame, mtx, dist)

        return frame
    return None

def calculate_trajectory(ball_positions):
    if len(ball_positions) < 2:
        return None  

    x_coords = [pos[0] for pos in ball_positions]
    y_coords = [pos[1] for pos in ball_positions]
    coefficients = np.polyfit(x_coords, y_coords, 2)
    return coefficients  

def analyze_trajectory(trajectory, hoop_position):
    a, b, c = trajectory

    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c

    if vertex_y < HOOP_HEIGHT:
        return "Increase launch angle to clear the hoop."
    
    intersection_x = (HOOP_HEIGHT - c) / a  
    if intersection_x < 0 or intersection_x > FRAME_WIDTH:
        return "Adjust your aim; the ball trajectory is too low or too high."

    hoop_center_x = (hoop_position[0] + hoop_position[2]) / 2 
    if intersection_x < hoop_center_x - (HOOP_WIDTH / 2):
        return "Aim more to the right."
    elif intersection_x > hoop_center_x + (HOOP_WIDTH / 2):
        return "Aim more to the left."
    
    return "Good shot trajectory!"

# Main loop
ball_positions = []  
hoop_position = None  

while True:
    frame = capture_frame()
    if frame is None:
        continue

    results = model(frame)

    ball_positions.clear() 
    for result in results:
        for detection in result.boxes:
            if detection.cls == 'basketball':
                ball_position = (detection.xyxy[0], detection.xyxy[1])  
                ball_positions.append(ball_position)
            elif detection.cls == 'hoop':
                hoop_position = detection.xyxy 

    if len(ball_positions) >= 2 and hoop_position is not None:
        trajectory = calculate_trajectory(ball_positions)

        if trajectory is not None:
            feedback = analyze_trajectory(trajectory, hoop_position)
            print(feedback)  

    cv2.imshow('Basketball Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
