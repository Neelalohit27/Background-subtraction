import numpy as np
import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture('C:/Users/lohit/OneDrive/Desktop/Background-subtraction/Background-subtraction/cars.mp4')
  # Make sure this path is correct
images = []

cv2.namedWindow('tracker')
cv2.createTrackbar('val', 'tracker', 50, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or can't read the frame.")
        break

    dim = (500, 500)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    images.append(frame_gray)

    if len(images) == 50:
        images.pop(0)

    image = np.array(images)

    val = cv2.getTrackbarPos('val', 'tracker')

    image = np.mean(image, axis=0).astype(np.uint8)

    # Show background image
    cv2.imshow('background', image)

    # Calculate foreground
    foreground_image = cv2.absdiff(frame_gray, image)

    # Threshold based on tracker
    a = np.array([0], np.uint8)
    b = np.array([255], np.uint8)

    img = np.where(foreground_image > val, frame_gray, a)

    # Show results
    cv2.imshow('image', frame)
    cv2.imshow('foreground', img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
