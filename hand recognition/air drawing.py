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



            # Get the index finger tip coordinates.
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)



            # Optionally, check if only index finger is up and other fingers down to toggle drawing
            # For simplicity, we draw always when the hand is detected.



            # Draw circle on fingertip for feedback.
            cv2.circle(frame, (x, y), 10, (0, 255, 255), cv2.FILLED)



            if prev_x is None and prev_y is None:
                prev_x, prev_y = x, y



            # Draw line on canvas from previous point to current point.
            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), thickness=5)



            prev_x, prev_y = x, y



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
