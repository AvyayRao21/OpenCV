import cv2
import numpy as np
import serial
import time
from collections import defaultdict

# Expanded color ranges in HSV (Hue, Saturation, Value)
color_ranges = {
    "Red": [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
    "Green": [(40, 100, 100), (80, 255, 255)],
    "Blue": [(100, 100, 100), (140, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Purple": [(140, 100, 100), (160, 255, 255)],
    "Orange": [(10, 100, 100), (20, 255, 255)],
    "Pink": [(160, 100, 100), (170, 255, 255)],
    "Cyan": [(85, 100, 100), (95, 255, 255)],
}

# Corresponding RGB values for the colors
rgb_values = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Purple": (128, 0, 128),
    "Orange": (255, 165, 0),
    "Pink": (255, 192, 203),
    "Cyan": (0, 255, 255),
}

def detect_dominant_color(hsv_frame):
    color_counts = defaultdict(int)
    
    for color_name, ranges in color_ranges.items():
        if color_name == "Red":  # Special case for red (2 ranges)
            lower1 = np.array(ranges[0], dtype=np.uint8)
            upper1 = np.array(ranges[1], dtype=np.uint8)
            lower2 = np.array(ranges[2], dtype=np.uint8)
            upper2 = np.array(ranges[3], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv_frame, lower1, upper1)
            mask2 = cv2.inRange(hsv_frame, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array(ranges[0], dtype=np.uint8)
            upper = np.array(ranges[1], dtype=np.uint8)
            mask = cv2.inRange(hsv_frame, lower, upper)
        
        color_counts[color_name] = cv2.countNonZero(mask)
    
    dominant_color = max(color_counts, key=color_counts.get)
    return dominant_color if color_counts[dominant_color] > 500 else "Unknown"

def main():
    # Set up serial communication with Arduino
    ser = serial.Serial('COM3', 9600)  # Change 'COM3' to your Arduino port
    time.sleep(2)  # Wait for the connection to establish

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dominant_color = detect_dominant_color(hsv_frame)
        
        # Send RGB value to Arduino
        if dominant_color in rgb_values:
            rgb = rgb_values[dominant_color]
            ser.write(f"{rgb[0]},{rgb[1]},{rgb[2]}\n".encode())
        
        # Visualization: Highlight the detected color
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Dominant Color: {dominant_color}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0) if dominant_color == "Unknown" else (255, 255, 255), 
                    2, cv2.LINE_AA)
        
        cv2.imshow("Advanced Color Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
