import cv2
import numpy as np
# Load template and target images template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('target.png', cv2.IMREAD_GRAYSCALE)
# Perform template matching result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
# Find locations above threshold
threshold = 0.7
locations = np.where(result >= threshold)
matches = list(zip(*locations[::-1])) # Convert to (x, y) format
