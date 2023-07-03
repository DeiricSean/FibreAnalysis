import os
from generate_test import test_generation_is_deterministic

if __name__ == '__main__':

    current_directory = os.getcwd()
    print('Current Directory', current_directory)

    background_images = os.path.join(current_directory, 'Data', 'Backgrounds', '')
    sample_images = os.path.join(current_directory, 'Data', 'synth', 'images', '')
    sample_masks = os.path.join(current_directory, 'Data', 'synth', 'masks', '')
    sample_labels = os.path.join(current_directory, 'Data', 'synth', 'labels', '')
    print(sample_masks)
    print(sample_images)
    
    print("Starting Image Creation")
    
    NumImages = 15
    test_generation_is_deterministic(sample_images , sample_masks , sample_labels, background_images, NumImages)
    print("Image Creation complete")