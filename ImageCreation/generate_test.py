# Source https://github.com/hughfdjackson/fluorescent-fibre-counting.git


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


def test_generation_is_deterministic(image_destination, mask_destination):
    test_seed = 120

    seed(test_seed)
    data1, label1, count1 = training_set(10, Config())

    for i, (image, mask) in enumerate(zip(data1, label1), 1):
        
        #Image.fromarray(image.squeeze(), mode="L").save(f"image{i}.png")
        Image.fromarray(image.squeeze(), mode="L").save(f"{image_destination}image{i}.png")
        Image.fromarray(mask.squeeze(), mode="L").save(f"{mask_destination}mask{i}.png")
       

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
