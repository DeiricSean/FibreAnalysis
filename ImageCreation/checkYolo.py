# Quick Visual validation to see if the yolo labels overlay the correct areas of the image file

import numpy as np
import os
import cv2

current_directory = os.getcwd()
sample_images = os.path.join(current_directory, 'Data', 'synth', 'images', '')
label_images = os.path.join(current_directory, 'Data', 'synth', 'labels', '')



image_path = os.path.join(sample_images, "image_2023-06-22_20-26-39-590405.png")
label_path = os.path.join(label_images, "label_2023-06-22_20-26-39-590405.txt")

# Load the original image and YOLO format label
image = cv2.imread(image_path)
#label_path = "yolo_label.txt"

# Read the YOLO format label
with open(label_path, "r") as file:
     lines = file.readlines()

# Process each line in the YOLO format label
for line in lines:
    line = line.strip().split()
  
    class_id = int(line[0])
    segment_points = list(map(float, line[1:]))

    # Reshape segment points into a list of (x, y) coordinates
    num_points = len(segment_points) // 2
    points = [(int(segment_points[i] * image.shape[1]), int(segment_points[i + 1] * image.shape[0])) for i in range(0, len(segment_points), 2)]

    # Draw a filled polygon on the original image to highlight the segment
    cv2.fillPoly(image, [np.array(points)], (0, 0, 255))

# Display the image with highlighted segments
cv2.imshow("Image with Segments", image)
cv2.waitKey(0)
cv2.destroyAllWindows()