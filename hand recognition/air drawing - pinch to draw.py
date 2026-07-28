import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,         # Track one hand.
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables for drawing.
drawing = False
prev_x, prev_y = None, None
canvas = None

def main():
    global drawing, prev_x, prev_y, canvas

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera")
        return

    # Create a black canvas of the same size as the frame.
    canvas = np.zeros_like(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for natural selfie view.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get the index finger tip and thumb tip coordinates.
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_x = int(index_finger_tip.x * w)
            index_y = int(index_finger_tip.y * h)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Draw circle on fingertip and thumb tip for feedback.
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)

            # Calculate the distance between index finger tip and thumb tip.
            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # Set a threshold for drawing (you can adjust this value).
            threshold = 20  # Adjust this value based on your needs.

            if distance < threshold:
                # If fingers are close enough, start drawing.
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = index_x, index_y

                # Draw line on canvas from previous point to current point.
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), thickness=5)
                prev_x, prev_y = index_x, index_y
            else:
                # If fingers are not close, lift the pen.
                prev_x, prev_y = None, None

            # Draw hand landmarks on the frame.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # No hand detected, reset previous point.
            prev_x, prev_y = None, None

        # Overlay the canvas on the frame.
        frame = cv2.addWeighted(frame, 1, canvas, 0.6, 0)

        # Instructions
        cv2.putText(frame, "Press 'c' to clear, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Virtual Drawing", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas = np.zeros_like(frame)  # Clear drawing
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
