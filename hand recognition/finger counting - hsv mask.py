import cv2
import numpy as np



def preprocess_roi(roi):
    """Preprocess ROI to create a clean mask for hand"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)



    # Define skin color range in HSV
    # values from HSV tuner
    lower_skin = np.array([0, 50, 142], dtype=np.uint8)
    upper_skin = np.array([179, 255, 255], dtype=np.uint8)



    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)



    # Apply a series of morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)



    # Smooth the mask
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    return mask



def count_fingers(max_contour, drawing):
    """Count fingers using convexity defects analysis"""
    hull = cv2.convexHull(max_contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0



    defects = cv2.convexityDefects(max_contour, hull)



    if defects is None:
        return 0



    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])



        # Calculate lengths of triangle sides
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))



        # Calculate angle using cosine rule (in degrees)
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / np.pi



        # Filter defects by angle and depth (d)
        if angle <= 90 and d > 10000:
            finger_count += 1
            cv2.circle(drawing, far, 8, [255, 0, 0], -1)
        # Draw lines between start and end points
        cv2.line(drawing, start, end, [0, 255, 0], 2)
    return finger_count



def main():
    cap = cv2.VideoCapture(0)
    # Set camera resolution higher for better detail
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



    if not cap.isOpened():
        print("Cannot open camera")
        return



    # Region of interest coordinates (adjust if needed)
    roi_top, roi_bottom = 100, 400
    roi_right, roi_left = 350, 650



    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break



        frame = cv2.flip(frame, 1)  # Mirror image



        # Draw ROI rectangle on frame where user places hand
        cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]



        # Preprocess ROI to get mask
        mask = preprocess_roi(roi)



        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 4000:
                # Approximate contour for smoothing
                epsilon = 0.01 * cv2.arcLength(max_contour, True)
                approx = cv2.approxPolyDP(max_contour, epsilon, True)



                # Draw contour on ROI
                cv2.drawContours(roi, [approx], -1, (0, 0, 255), 3)



                # Count fingers
                fingers = count_fingers(approx, roi)



                # Display finger count (fingers+1 as defect count is one less than finger count)
                cv2.putText(frame, f"Fingers: {fingers + 1}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            else:
                cv2.putText(frame, "Place hand inside box", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Waiting for hand...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        # Show the windows
        cv2.imshow("Hand Detection", frame)
        cv2.imshow("Mask", mask)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


