import cv2
import numpy as np
import mediapipe as mp
import time



# Initialize MediaPipe hands for up to 2 hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)



# Keyboard layout
keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]



key_fill_color = (200, 200, 200)  # light gray
key_fill_alpha = 0.3
key_hover_color = (255, 255, 255)
key_hover_alpha = 0.5
key_press_color = (0, 255, 0)
key_press_alpha = 0.7



last_key_press_time = 0
key_delay = 1.0  # seconds debounce delay



typed_text = ""



def draw_transparent_rect(frame, top_left, bottom_right, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



def draw_keyboard(frame, width, height, hover_keys=None, pressed_keys=None):
    key_width = width // 10
    key_height = height // 6
    key_positions = []



    if hover_keys is None:
        hover_keys = set()
    if pressed_keys is None:
        pressed_keys = set()



    for i, row in enumerate(keyboard_layout):
        row_y = i * key_height + height // 4
        row_x_offset = (width - key_width * len(row)) // 2
        for j, key in enumerate(row):
            x = row_x_offset + j * key_width
            y = row_y
            top_left = (x, y)
            bottom_right = (x + key_width, y + key_height)



            if key in pressed_keys:
                draw_transparent_rect(frame, top_left, bottom_right, key_press_color, key_press_alpha)
            elif key in hover_keys:
                draw_transparent_rect(frame, top_left, bottom_right, key_hover_color, key_hover_alpha)
            else:
                draw_transparent_rect(frame, top_left, bottom_right, key_fill_color, key_fill_alpha)



            cv2.rectangle(frame, top_left, bottom_right, (170,170,170), 2)



            font_scale = 1.5
            font_thickness = 3
            text_size, _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = x + (key_width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2 - 5
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)



            key_positions.append((key, top_left, bottom_right))



    # Space bar
    space_width = key_width * 5
    space_height = key_height
    space_x = (width - space_width) // 2
    space_y = height // 4 + 3 * key_height + 10
    space_top_left = (space_x, space_y)
    space_bottom_right = (space_x + space_width, space_y + space_height)



    if "SPACE" in pressed_keys:
        draw_transparent_rect(frame, space_top_left, space_bottom_right, key_press_color, key_press_alpha)
    elif "SPACE" in hover_keys:
        draw_transparent_rect(frame, space_top_left, space_bottom_right, key_hover_color, key_hover_alpha)
    else:
        draw_transparent_rect(frame, space_top_left, space_bottom_right, key_fill_color, key_fill_alpha)



    cv2.rectangle(frame, space_top_left, space_bottom_right, (170,170,170), 2)
    space_text = "SPACE"
    font_scale = 1.5
    font_thickness = 3
    text_size, _ = cv2.getTextSize(space_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = space_x + (space_width - text_size[0]) // 2
    text_y = space_y + (space_height + text_size[1]) // 2 - 5
    cv2.putText(frame, space_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)



    # Backspace key
    backspace_width = key_width * 2
    backspace_height = key_height
    backspace_x = width - backspace_width - 15
    backspace_y = height // 20
    backspace_top_left = (backspace_x, backspace_y)
    backspace_bottom_right = (backspace_x + backspace_width, backspace_y + backspace_height)



    if "BACKSPACE" in pressed_keys:
        draw_transparent_rect(frame, backspace_top_left, backspace_bottom_right, key_press_color, key_press_alpha)
    elif "BACKSPACE" in hover_keys:
        draw_transparent_rect(frame, backspace_top_left, backspace_bottom_right, key_hover_color, key_hover_alpha)
    else:
        draw_transparent_rect(frame, backspace_top_left, backspace_bottom_right, key_fill_color, key_fill_alpha)



    cv2.rectangle(frame, backspace_top_left, backspace_bottom_right, (170,170,170), 2)
    backspace_text = "<-"
    font_scale = 1.5
    font_thickness = 3
    text_size, _ = cv2.getTextSize(backspace_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = backspace_x + (backspace_width - text_size[0]) // 2
    text_y = backspace_y + (backspace_height + text_size[1]) // 2 - 5
    cv2.putText(frame, backspace_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)



    key_positions.append(("SPACE", space_top_left, space_bottom_right))
    key_positions.append(("BACKSPACE", backspace_top_left, backspace_bottom_right))
    return key_positions



def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))



def is_pinch(thumb_tip, index_tip, threshold=40):
    return distance(thumb_tip, index_tip) < threshold



def is_pinky_extended(hand_landmarks, width, height):
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    tip_y = pinky_tip.y * height
    pip_y = pinky_pip.y * height
    return tip_y < pip_y



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
        if distance(tip, wrist) > 0.2 * height:
            extended += 1
    return extended == 5



def main():
    global last_key_press_time, typed_text



    cap = cv2.VideoCapture(0)



    pressed_keys = set()
    hover_keys = set()



    while True:
        ret, frame = cap.read()
        if not ret:
            break



        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape



        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



        # For hovered and pressed keys in this frame
        frame_hover_keys = set()
        frame_pressed_keys = set()



        # To detect which key each hand is on
        index_finger_positions = []



        # Prepare for debouncing and pressed key logic
        current_time = time.time()
        can_press = (current_time - last_key_press_time) > key_delay



        key_positions = draw_keyboard(frame, width, height, hover_keys=hover_keys, pressed_keys=pressed_keys)



        # Prepare mapping of keys to rectangles for hit-test
        key_rect_map = {}
        for key, tl, br in key_positions:
            key_rect_map[key] = (tl, br)



        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmarks to pixel coords
                wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width,
                         hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                index_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width,
                             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                thumb_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width,
                             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)



                # Draw hand landmarks
                for lm in hand_landmarks.landmark:
                    x_lm = int(lm.x * width)
                    y_lm = int(lm.y * height)
                    cv2.circle(frame, (x_lm, y_lm), 6, (0, 255, 0), -1)



                # Gesture detections
                open_palm = is_open_palm(hand_landmarks, width, height)
                pinky_ext = is_pinky_extended(hand_landmarks, width, height)
                pinch = is_pinch(thumb_tip, index_tip)



                # Detect hovered key by index finger location
                ix, iy = int(index_tip[0]), int(index_tip[1])
                cv2.circle(frame, (ix, iy), 12, (255, 0, 255), 2)



                hovered_key = None
                for key, (x1,y1), (x2,y2) in key_positions:
                    if x1 <= ix <= x2 and y1 <= iy <= y2:
                        hovered_key = key
                        frame_hover_keys.add(key)
                        break



                # Decide key press for this hand
                if can_press:
                    # Priority: space (open palm), backspace (pinky), letter (pinch)
                    if open_palm:
                        frame_pressed_keys.add("SPACE")
                    elif pinky_ext and not open_palm:
                        frame_pressed_keys.add("BACKSPACE")
                    elif pinch and hovered_key is not None and hovered_key not in ["SPACE", "BACKSPACE"]:
                        frame_pressed_keys.add(hovered_key)



        # Update global pressed keys and typed_text if any new presses
        if frame_pressed_keys and can_press:
            for key in frame_pressed_keys:
                if key == "SPACE":
                    typed_text += " "
                    print("Key pressed: SPACE")
                elif key == "BACKSPACE":
                    if len(typed_text) > 0:
                        typed_text = typed_text[:-1]
                    print("Key pressed: BACKSPACE")
                else:
                    # If Q pressed, exit program
                    if key.upper() == "Q":
                        print("Exiting program via Q key.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    typed_text += key
                    print(f"Key pressed: {key}")
            last_key_press_time = current_time
            pressed_keys = frame_pressed_keys.copy()
        else:
            pressed_keys.clear()



        hover_keys = frame_hover_keys.copy()



        # Typed text display box
        box_height = 70
        cv2.rectangle(frame, (0,0), (width, box_height), (50,50,50), -1)
        display_text = typed_text[-35:]
        cv2.putText(frame, display_text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)



        cv2.imshow("Virtual Keyboard - Two Hand", frame)



        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


