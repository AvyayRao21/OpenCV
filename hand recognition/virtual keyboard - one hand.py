import cv2
import numpy as np
import mediapipe as mp
import time



# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)



# Define keyboard layout - rows with keys, space bar will be added below separately
keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]



# Transparent key colors
key_fill_color = (200, 200, 200)  # light gray
key_fill_alpha = 0.3  # transparency for keys (0 fully transparent, 1 opaque)
key_hover_color = (255, 255, 255)  # white for hovered key background
key_hover_alpha = 0.5
key_press_color = (0, 255, 0)  # green for pressed key
key_press_alpha = 0.7



# Last time a key was pressed to implement debounce
last_key_press_time = 0
key_delay = 1.0  # seconds



typed_text = ""



# Utilities for drawing transparent rectangles on frame
def draw_transparent_rect(frame, top_left, bottom_right, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



def draw_keyboard(frame, width, height, hover_key=None, pressed_key=None):
    key_width = width // 10
    key_height = height // 6
    key_positions = []  # Store positions and keys for hit test



    for i, row in enumerate(keyboard_layout):
        row_y = i * key_height + height // 4
        row_x_offset = (width - key_width * len(row)) // 2  # Center each row horizontally
        for j, key in enumerate(row):
            x = row_x_offset + j * key_width
            y = row_y
            top_left = (x, y)
            bottom_right = (x + key_width, y + key_height)



            # Choose color based on state
            if pressed_key == key:
                draw_transparent_rect(frame, top_left, bottom_right, key_press_color, key_press_alpha)
            elif hover_key == key:
                draw_transparent_rect(frame, top_left, bottom_right, key_hover_color, key_hover_alpha)
            else:
                draw_transparent_rect(frame, top_left, bottom_right, key_fill_color, key_fill_alpha)



            # Draw key border
            cv2.rectangle(frame, top_left, bottom_right, (170, 170, 170), 2)



            # Draw key text in black
            font_scale = 1.5
            font_thickness = 3
            text_size, _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = x + (key_width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2 - 5
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)



            key_positions.append((key, top_left, bottom_right))



    # Draw space bar centered below keys
    space_width = key_width * 5
    space_height = key_height
    space_x = (width - space_width) // 2
    space_y = height // 4 + 3 * key_height + 10
    space_top_left = (space_x, space_y)
    space_bottom_right = (space_x + space_width, space_y + space_height)



    if pressed_key == "SPACE":
        draw_transparent_rect(frame, space_top_left, space_bottom_right, key_press_color, key_press_alpha)
    else:
        draw_transparent_rect(frame, space_top_left, space_bottom_right, key_fill_color, key_fill_alpha)
    cv2.rectangle(frame, space_top_left, space_bottom_right, (170, 170, 170), 2)
    space_text = "SPACE"
    font_scale = 1.5
    font_thickness = 3
    text_size, _ = cv2.getTextSize(space_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = space_x + (space_width - text_size[0]) // 2
    text_y = space_y + (space_height + text_size[1]) // 2 - 5
    cv2.putText(frame, space_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)



    # Draw backspace key on top right corner
    backspace_width = key_width * 2
    backspace_height = key_height
    backspace_x = width - backspace_width - 15
    backspace_y = height // 20
    backspace_top_left = (backspace_x, backspace_y)
    backspace_bottom_right = (backspace_x + backspace_width, backspace_y + backspace_height)



    if pressed_key == "BACKSPACE":
        draw_transparent_rect(frame, backspace_top_left, backspace_bottom_right, key_press_color, key_press_alpha)
    else:
        draw_transparent_rect(frame, backspace_top_left, backspace_bottom_right, key_fill_color, key_fill_alpha)
    cv2.rectangle(frame, backspace_top_left, backspace_bottom_right, (170, 170, 170), 2)
    backspace_text = "<-"
    font_scale = 1.5
    font_thickness = 3
    text_size, _ = cv2.getTextSize(backspace_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = backspace_x + (backspace_width - text_size[0]) // 2
    text_y = backspace_y + (backspace_height + text_size[1]) // 2 - 5
    cv2.putText(frame, backspace_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)



    # Return all key positions for hit testing (keys + special keys)
    key_positions.append(("SPACE", space_top_left, space_bottom_right))
    key_positions.append(("BACKSPACE", backspace_top_left, backspace_bottom_right))
    return key_positions



# Calculate distance between two points
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))



# Detect pinch gesture (distance between thumb tip and index finger tip below threshold)
def is_pinch(thumb_tip, index_tip, threshold=40):
    return distance(thumb_tip, index_tip) < threshold



# Detect if pinky finger is extended (using landmark positions)
def is_pinky_extended(hand_landmarks, width, height):
    # Pinky tip and Pinky pip (middle joint)
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    # Convert to screen coords
    tip_y = pinky_tip.y * height
    pip_y = pinky_pip.y * height



    # If pinky tip is above pip, it's extended
    return tip_y < pip_y



# Detect open palm (all five fingers extended)
def is_open_palm(hand_landmarks, width, height):
    wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width,
             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
    extended = 0
    fingers = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP,
               mp_hands.HandLandmark.THUMB_TIP]
    for f in fingers:
        tip = (hand_landmarks.landmark[f].x * width,
               hand_landmarks.landmark[f].y * height)
        if distance(tip, wrist) > 0.2 * height:  # threshold proportional to frame height
            extended += 1
    return extended == 5



def main():
    global last_key_press_time, typed_text



    cap = cv2.VideoCapture(0)



    pressed_key = None
    hover_key = None



    while True:
        ret, frame = cap.read()
        if not ret:
            break



        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape



        frame_bgr = frame.copy()



        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



        index_tip_pos = None
        thumb_tip_pos = None
        detected_open_palm = False
        pinky_extended = False



        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]



            wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width,
                     hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
            index_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width,
                         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
            thumb_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width,
                         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)



            index_tip_pos = index_tip
            thumb_tip_pos = thumb_tip



            # Draw hand landmarks for visual feedback
            for lm in hand_landmarks.landmark:
                x_lm = int(lm.x * width)
                y_lm = int(lm.y * height)
                cv2.circle(frame, (x_lm, y_lm), 6, (0, 255, 0), -1)



            detected_open_palm = is_open_palm(hand_landmarks, width, height)
            pinky_extended = is_pinky_extended(hand_landmarks, width, height)



            pinch = is_pinch(thumb_tip, index_tip)
        else:
            pinch = False
            detected_open_palm = False
            pinky_extended = False



        key_positions = draw_keyboard(frame, width, height, hover_key=hover_key, pressed_key=pressed_key)



        hovered_key = None



        if index_tip_pos is not None:
            ix, iy = int(index_tip_pos[0]), int(index_tip_pos[1])
            cv2.circle(frame, (ix, iy), 12, (255, 0, 255), 2)
            for key, top_left, bottom_right in key_positions:
                x1, y1 = top_left
                x2, y2 = bottom_right
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    hovered_key = key
                    break



        current_time = time.time()
        if current_time - last_key_press_time > key_delay:
            if detected_open_palm:
                # SPACE bar on open palm
                typed_text += " "
                print("Key pressed: SPACE")
                last_key_press_time = current_time
                pressed_key = "SPACE"
            elif pinky_extended and not detected_open_palm:
                # BACKSPACE on pinky extended except space
                if len(typed_text) > 0:
                    typed_text = typed_text[:-1]
                print("Key pressed: BACKSPACE")
                last_key_press_time = current_time
                pressed_key = "BACKSPACE"
            elif pinch and hovered_key is not None and hovered_key not in ["SPACE", "BACKSPACE"]:
                if hovered_key.upper() == 'Q':
                    print("Exiting program via Q key.")
                    break
                typed_text += hovered_key
                print(f"Key pressed: {hovered_key}")
                last_key_press_time = current_time
                pressed_key = hovered_key
            else:
                pressed_key = None
        else:
            pressed_key = None



        hover_key = hovered_key



        # Draw typed text box on top
        box_height = 70
        cv2.rectangle(frame, (0, 0), (width, box_height), (50, 50, 50), -1)
        # Clip typed_text to fit width visually (max ~35 chars approx)
        display_text = typed_text[-35:]
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)



        cv2.imshow("Virtual Keyboard", frame)



        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


