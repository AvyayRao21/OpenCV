import cv2
import mediapipe as mp
import numpy as np



# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils



# Finger tip landmarks (index, middle, ring, pinky, thumb)
FINGER_TIPS = [mp_hands.HandLandmark.THUMB_TIP,
               mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]



# Finger pip landmarks for comparison (for thumb different landmark)
FINGER_PIPS = [mp_hands.HandLandmark.THUMB_IP,
               mp_hands.HandLandmark.INDEX_FINGER_PIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
               mp_hands.HandLandmark.RING_FINGER_PIP,
               mp_hands.HandLandmark.PINKY_PIP]



def count_fingers(hand_landmarks, image_width, image_height):
    fingers_open = 0



    # Wrist coordinate for reference
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]



    # Thumb: compare tip and IP landmark in x direction (for right hand)
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    thumb_ip = hand_landmarks.landmark[FINGER_PIPS[0]]
    # Thumb bends differently so check horizontal position depending on hand (left/right)
    if thumb_tip.x < thumb_ip.x:
        fingers_open += 1



    # Other fingers: check if tip is above pip in y axis
    for i in range(1, 5):
        tip = hand_landmarks.landmark[FINGER_TIPS[i]]
        pip = hand_landmarks.landmark[FINGER_PIPS[i]]
        if tip.y < pip.y:
            fingers_open += 1



    return fingers_open



def main():
    cap = cv2.VideoCapture(0)



    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:



        while True:
            ret, frame = cap.read()
            if not ret:
                break



            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            finger_count = 0



            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                finger_count = count_fingers(hand_landmarks, frame.shape[1], frame.shape[0])



                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)



            # Display finger count
            cv2.rectangle(frame, (0, 0), (150, 70), (0, 0, 0), -1)
            cv2.putText(frame, f'Fingers: {finger_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)



            cv2.imshow('Finger Counting', frame)



            if cv2.waitKey(1) & 0xFF == 27:
                break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
