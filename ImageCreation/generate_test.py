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



def test_generation_is_deterministic():
    test_seed = 120

    seed(test_seed)
    data1, label1, count1 = training_set(10, Config())

 #   seed(test_seed)
 #   data2, label2, count2 = training_set(10, Config())

 #   np.array_equal(data1, data2)
 #   np.array_equal(label1, label2)
 #   np.array_equal(count1, count2)


# Create a PIL Image object
    image = Image.fromarray(data1[0].squeeze(), mode="L")

# Save the image
    image.save("image  test test .png")  # Save the image as PNG file


    # Reshape the array to 2D
    mask_array_2d = label1[0].squeeze()

    # Create a PIL Image object
    mask_image = Image.fromarray(mask_array_2d * 255, mode="L")

    # Save the mask as a PNG file
    mask_image.save("mask.png")  # Save the mask as a PNG file