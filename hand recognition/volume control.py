import cv2
import mediapipe as mp
import numpy as np
import math
import platform



# For Windows volume control
if platform.system() == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



def set_volume(volume_percent):
    """
    Set system volume as percent (0 to 100).
    Currently implemented for Windows using pycaw.
    """
    system = platform.system()
    vol = np.clip(volume_percent, 0, 50) / 100.0  # normalize 0-1



    if system == "Windows":
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(vol, None)
    else:
        # For other OS, you can implement using OS-specific system calls or libraries
        # For macOS you might use osascript, for Linux amixer or pactl commands.
        pass



def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)



    mp_draw = mp.solutions.drawing_utils



    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return



    # Volume control range in cm (approx), adjust experimentally
    MIN_DIST = 15    # Minimum distance for volume = 0%
    MAX_DIST = 150   # Maximum distance for volume = 100%



    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break



        frame = cv2.flip(frame, 1)  # Mirror img
        h, w, c = frame.shape



        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)



        volume_percent = None



        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]



            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)



            # Extract thumb tip and index finger tip positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]



            # Convert normalized coords to pixels
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)



            # Draw circles on thumb and index finger tips
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)



            # Calculate distance in pixels between thumb and index
            dist = math.hypot(index_x - thumb_x, index_y - thumb_y)



            # Map distance to volume percent
            volume_percent = np.interp(dist, [MIN_DIST, MAX_DIST], [0, 100])
            volume_percent = int(np.clip(volume_percent, 0, 100))



            # Set volume
            set_volume(volume_percent)



            # Draw volume bar and text
            bar_x_start, bar_y_start = 50, 400
            bar_height = 300
            bar_width = 30
            vol_bar = np.interp(volume_percent, [0, 100], [bar_y_start + bar_height, bar_y_start])



            cv2.rectangle(frame, (bar_x_start, bar_y_start), (bar_x_start + bar_width, bar_y_start + bar_height), (180, 180, 180), 3)
            cv2.rectangle(frame, (bar_x_start, int(vol_bar)), (bar_x_start + bar_width, bar_y_start + bar_height), (0, 255, 0), cv2.FILLED)



            cv2.putText(frame, f'Vol: {volume_percent}%', (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



        else:
            # No hand detected
            cv2.putText(frame, "Show your hand with thumb and index finger visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



        cv2.imshow("Hand Volume Control", frame)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


