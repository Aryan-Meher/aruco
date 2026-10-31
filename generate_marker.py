
import cv2
import cv2.aruco as aruco
import numpy as np

# Create ArUco dictionary
dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

# Generate marker ID 23
marker = aruco.generateImageMarker(
    dictionary, 23, 600, borderBits=1
)

# Create a white background
canvas = np.full((800, 800), 255, dtype=np.uint8)

# Place marker in the center
canvas[100:700, 100:700] = marker

# Save marker
cv2.imwrite("marker23.png", canvas)

print("Marker generated successfully!")
