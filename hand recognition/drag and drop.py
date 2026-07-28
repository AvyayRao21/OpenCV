import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Hands.
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Object properties
object_radius = 40
object_color = (0, 165, 255)  # Orange color
canvas = None

# Initial random position for the object - will be set after first frame capture
obj_x, obj_y = None, None

# Dragging state
dragging = False
offset_x, offset_y = 0, 0

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    global canvas, obj_x, obj_y, dragging, offset_x, offset_y

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return

    h, w, _ = frame.shape

    # Initialize object position randomly within frame bounds with padding for radius
    obj_x = random.randint(object_radius, w - object_radius)
    obj_y = random.randint(object_radius, h - object_radius)

    canvas = np.zeros_like(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)  # Mirror image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw the draggable object on canvas
        if canvas is None:
            canvas = np.zeros_like(frame)
        else:
            canvas[:] = 0  # Clear canvas for each frame

        cv2.circle(frame, (obj_x, obj_y), object_radius, object_color, -1)
        cv2.circle(frame, (obj_x, obj_y), object_radius, (0, 140, 255), 3)  # outline for better visibility

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get index finger tip and thumb tip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate distance between thumb and index finger
            pinch_distance = distance((index_x, index_y), (thumb_x, thumb_y))
            pinch_threshold = 40  # Tuned threshold for pinch detection

            # Draw circles on finger tips
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), cv2.FILLED)
            # Draw line between thumb and index finger tips
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)

            # Check if pinch is detected near the object
            finger_center_x = (index_x + thumb_x) // 2
            finger_center_y = (index_y + thumb_y) // 2
            dist_to_object = distance((finger_center_x, finger_center_y), (obj_x, obj_y))

            if pinch_distance < pinch_threshold and dist_to_object < object_radius + 20:
                if not dragging:
                    # Start dragging, calculate offset to avoid jump when dragging starts
                    offset_x = obj_x - finger_center_x
                    offset_y = obj_y - finger_center_y
                    dragging = True
                # Update object position with finger position plus offset
                obj_x = finger_center_x + offset_x
                obj_y = finger_center_y + offset_y

                # Clamp position inside frame bounds
                obj_x = max(object_radius, min(w - object_radius, obj_x))
                obj_y = max(object_radius, min(h - object_radius, obj_y))
            else:
                dragging = False

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            dragging = False

        # Overlay instructions
        cv2.putText(frame, "Pinch (thumb+index) object to drag it. Release to drop.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit.", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Virtual Drag and Drop", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

