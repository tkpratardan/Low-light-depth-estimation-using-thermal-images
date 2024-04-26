import cv2
import numpy as np

def median_filter(image):
    # Create a 3x3 kernel
    kernel = np.ones((3, 3), np.uint8)
    # Apply median filtering
    filtered_image = cv2.medianBlur(image, 3)
    return filtered_image

# Load an image
image = cv2.imread('image.jpg')

# Apply median filtering
filtered_image = median_filter(image)

