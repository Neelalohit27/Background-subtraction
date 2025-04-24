import cv2
import numpy as np

def nothing(x):
    pass

# Create windows and trackbars
cv2.namedWindow('Controls')
cv2.createTrackbar('Learning Rate (x1000)', 'Controls', 20, 1000, nothing)  # alpha = 0.02 default
cv2.createTrackbar('Threshold', 'Controls', 25, 100, nothing)  # threshold = 2.5 default
cv2.createTrackbar('Init Variance', 'Controls', 100, 1000, nothing)  # Lower initial variance
cv2.createTrackbar('Blur Size', 'Controls', 3, 15, nothing)  # Gaussian blur kernel size

# Initialize video capture
cap = cv2.VideoCapture('C:/Users/lohit/OneDrive/Desktop/Background-subtraction/Background-subtraction/cars.mp4')

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Initialize the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame")
    exit()

# Initialize background model
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)
mean = frame_gray.astype(np.float32)
height, width = mean.shape
init_var = 100  # Initial variance (can be adjusted via trackbar)
var = np.ones((height, width), dtype=np.float32) * init_var

# Create masks for morphological operations
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break
    
    frame_count += 1
    
    # Get parameters from trackbars
    alpha = cv2.getTrackbarPos('Learning Rate (x1000)', 'Controls') / 1000.0
    threshold = cv2.getTrackbarPos('Threshold', 'Controls') / 10.0
    init_var = cv2.getTrackbarPos('Init Variance', 'Controls')
    blur_size = cv2.getTrackbarPos('Blur Size', 'Controls')
    if blur_size % 2 == 0:  # Ensure odd kernel size
        blur_size += 1
    
    # Preprocess frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_gray = cv2.GaussianBlur(frame_gray, (blur_size, blur_size), 0)
    
    # Calculate difference between current frame and mean background
    diff = frame_gray - mean
    
    # Calculate normalized distance
    normalized_diff = np.abs(diff) / np.sqrt(var + 1e-6)  # small epsilon to avoid division by zero
    
    # Create foreground mask based on threshold
    foreground_mask = (normalized_diff > threshold).astype(np.uint8) * 255
    
    # Apply morphological operations to reduce noise
    foreground_mask = cv2.erode(foreground_mask, kernel_erode, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel_dilate, iterations=1)
    foreground_mask = cv2.medianBlur(foreground_mask, 3)
    
    # Update equations for mean and variance with improved variance update
    new_mean = alpha * frame_gray + (1 - alpha) * mean
    new_var = alpha * (diff * diff) + (1 - alpha) * var
    
    # Update background model only for background pixels
    background_mask = (normalized_diff <= threshold)
    mean[background_mask] = new_mean[background_mask]
    var[background_mask] = new_var[background_mask]
    
    # Adaptive learning rate for static regions
    static_regions = (normalized_diff < threshold/2)
    mean[static_regions] = frame_gray[static_regions]
    var[static_regions] *= 0.99  # Slowly reduce variance in static regions
    
    # Visualizations
    vis_frame = frame.copy()
    vis_frame[foreground_mask == 0] = cv2.cvtColor(cv2.convertScaleAbs(mean), cv2.COLOR_GRAY2BGR)[foreground_mask == 0]
    
    # Show results
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    cv2.imshow('Background Model', cv2.convertScaleAbs(mean))
    cv2.imshow('Result', vis_frame)
    
    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
