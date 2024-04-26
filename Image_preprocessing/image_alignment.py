import cv2
import numpy as np

# Load the video capture device (e.g. a camera)
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()

# Convert the frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create a MOSSE filter for global motion estimation
mosse = cv2.TrackerMOSSE_create()

# Initialize the MOSSE filter with the first frame
mosse.init(gray1, (100, 100, 100, 100))  # adjust the rectangle to your needs

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Estimate the global motion parameter
    mosse.track(gray2)

    # Get the translation vector
    translation = mosse.get_state()

    # Apply the translation to the second frame
    aligned_frame = cv2.warpAffine(frame2, translation, (frame1.shape[1], frame1.shape[0]))

    # Estimate the dense optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Apply the optical flow to the aligned frame
    aligned_frame = cv2.remap(aligned_frame, flow, None, cv2.INTER_LINEAR)

    # Display the aligned frame
    cv2.imshow('Aligned Frame', aligned_frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device
cap.release()
cv2.destroyAllWindows()
