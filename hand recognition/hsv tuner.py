import cv2
import numpy as np

def tune_hsv_mask():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow('HSV Tuner')
    cv2.createTrackbar('LH', 'HSV Tuner', 0, 179, lambda x: None) # Lower Hue
    cv2.createTrackbar('LS', 'HSV Tuner', 40, 255, lambda x: None) # Lower Saturation
    cv2.createTrackbar('LV', 'HSV Tuner', 50, 255, lambda x: None) # Lower Value
    cv2.createTrackbar('UH', 'HSV Tuner', 20, 179, lambda x: None) # Upper Hue
    cv2.createTrackbar('US', 'HSV Tuner', 255, 255, lambda x: None) # Upper Saturation
    cv2.createTrackbar('UV', 'HSV Tuner', 255, 255, lambda x: None) # Upper Value

    # Set initial trackbar positions (you can change these to your last known good values)
    cv2.setTrackbarPos('LH', 'HSV Tuner', 0)
    cv2.setTrackbarPos('LS', 'HSV Tuner', 40)
    cv2.setTrackbarPos('LV', 'HSV Tuner', 50)
    cv2.setTrackbarPos('UH', 'HSV Tuner', 20)
    cv2.setTrackbarPos('US', 'HSV Tuner', 255)
    cv2.setTrackbarPos('UV', 'HSV Tuner', 255)

    print("\n--- HSV Tuning Guide ---")
    print("Adjust the sliders in the 'HSV Tuner' window.")
    print("Goal: Your hand should be PURE WHITE in the 'Mask' window.")
    print("Everything else (background) should be PURE BLACK.")
    print("Press 'q' to quit and see the final HSV values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Get current positions of trackbars
        lh = cv2.getTrackbarPos('LH', 'HSV Tuner')
        ls = cv2.getTrackbarPos('LS', 'HSV Tuner')
        lv = cv2.getTrackbarPos('LV', 'HSV Tuner')
        uh = cv2.getTrackbarPos('UH', 'HSV Tuner')
        us = cv2.getTrackbarPos('US', 'HSV Tuner')
        uv = cv2.getTrackbarPos('UV', 'HSV Tuner')

        lower_skin = np.array([lh, ls, lv], dtype=np.uint8)
        upper_skin = np.array([uh, us, uv], dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Optional: Apply some morphological operations to see their effect on the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2) # Dilate more to connect parts
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Show the mask (Crucial for tuning)
        cv2.imshow('Mask', mask)
        cv2.imshow('Original Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Final Tuned HSV Values ---")
    print(f"lower_skin = np.array([{lh}, {ls}, {lv}], dtype=np.uint8)")
    print(f"upper_skin = np.array([{uh}, {us}, {uv}], dtype=np.uint8)")

if __name__ == "__main__":
    tune_hsv_mask()
