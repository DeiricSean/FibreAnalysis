# Source https://github.com/hughfdjackson/fluorescent-fibre-counting.git

import os
from datetime import datetime
import cv2
from PIL import Image
from random import seed
from generate import (
    generate_training_example,
    training_set,
    Config,
    Fibre,
    gen_components
)


import numpy as np
import math

def get_contours( inboundMask ):

#for j in os.listdir(input_dir):
#    image_path = os.path.join(input_dir, j)
    # load the binary mask and get its contours
#    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, mask = cv2.threshold(inboundMask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(inboundMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
       # if cv2.contourArea(cnt) > 200:
        polygon = []
        for point in cnt:
            x, y = point[0]
            polygon.append(x / W)
            polygon.append(y / H)
        polygons.append(polygon)
    
    return polygons


def store_polygons(directory, file,  inboundPolygons):
    # print the polygons
#    with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
    with open('{}.txt'.format(os.path.join(directory, file)[:-4]), 'w') as f:
        for polygon in inboundPolygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))

        f.close()



def test_generation_is_deterministic(image_destination, mask_destination, label_destination):
    test_seed = 120

    seed(test_seed)
    data1, label1, count1 = training_set(10, Config())

    for i, (image, mask) in enumerate(zip(data1, label1), 1):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        #Image.fromarray(image.squeeze(), mode="L").save(f"image{i}.png")
        Image.fromarray(image.squeeze(), mode="L").save(f"{image_destination}image_{current_time}.png")
        #Image.fromarray(mask.squeeze(), mode="L").save(f"{mask_destination}mask_{current_time}.png")
        Image.fromarray(mask.squeeze(), mode="P").save(f"{mask_destination}mask_{current_time}.png")
        store_polygons(label_destination, f"label_{current_time}.txt",  get_contours(mask))

# Check if it contains a mask 

    # # Load the mask image
    # mask_image = Image.open("image_mask.png")

    # # Convert the image to a NumPy array
    # mask_array = np.array(mask_image)

    # # Check if any non-zero pixel values exist
    # contains_mask = np.any(mask_array != 0)

    # # Print the result
    # if contains_mask:
    #     print("The mask image contains a mask.")
    # else:
    #     print("The mask image does not contain a mask.")
