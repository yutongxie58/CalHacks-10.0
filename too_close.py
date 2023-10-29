import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

frame_width = int(cap.get(3))  # Capture frame width
frame_height = int(cap.get(4))  # Capture frame height
total_frame_area = frame_width * frame_height  # Total frame area


desired_coverage = 0.4
some_size_threshold = int(total_frame_area * desired_coverage)

while True:
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    movement_threshold = 5.5

    moving_pixels = np.where(mag > movement_threshold)

    if moving_pixels[0].any() and moving_pixels[1].any():
        contours, _ = cv2.findContours((mag > movement_threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the contour with the largest area
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box of the largest moving object

            # Check the size of the largest object as an indicator of proximity
            if w * h > some_size_threshold:
                print("TOO CLOSE!!!!")
                quit()

    cv2.imshow('Moving Object Detection', frame2)

    prvs = next

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
