import cv2
import numpy as np

# Creating video element
cap = cv2.VideoCapture('C:/Users/lohit/OneDrive/Desktop/Background-subtraction/Background-subtraction/cars.mp4')
# cap = cv2.VideoCapture('thunder2.mp4')

# Read the first frame to get the shape
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Could not read the video file or frame.")
    cap.release()
    exit()

# Getting shape of the frame
row, col, channel = frame.shape

# Initialising background and foreground
background = np.zeros([row, col], np.uint8)
foreground = np.zeros([row, col], np.uint8)

# Converting data type of integers 0 and 255 to uint8 type
a = np.uint8([255])
b = np.uint8([0])

# Creating kernel for removing noise
kernel = np.ones([3, 3], np.uint8)

while cap.isOpened():
    ret, frame1 = cap.read()
    if not ret:
        print("End of video or can't read the frame.")
        break

    frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Median approximation algorithm
    background = np.where(frame_gray > background, background + 1, background - 1)

    # Foreground = |background - current frame|
    foreground = cv2.absdiff(background, frame_gray)

    # Apply threshold
    foreground = np.where(foreground > 40, a, b)

    # Remove noise
    foreground = cv2.erode(foreground, kernel)
    foreground = cv2.dilate(foreground, kernel)

    # Get colored foreground
    color_foreground = cv2.bitwise_and(frame1, frame1, mask=foreground)

    cv2.imshow('Background', background)
    cv2.imshow('Foreground', color_foreground)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
