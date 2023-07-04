import os
from generate_test import test_generation_is_deterministic

if __name__ == '__main__':

    current_directory = os.getcwd()
    print('Current Directory', current_directory)

    # Create defined number of synthetic images
    for stageDirectory, numberImages in [("Train", 20), ("Val", 10), ("Test", 10)]:

        background_images = os.path.join(current_directory, 'Data', 'Backgrounds', '')
        sample_images = os.path.join(current_directory, 'Data', 'synth', stageDirectory,'images', '')
        sample_masks = os.path.join(current_directory, 'Data', 'synth',  stageDirectory, 'masks', '')
        #sample_labels = os.path.join(current_directory, 'Data', 'synth',  stageDirectory, 'labels', '') # labels done in secondary processing
        print(sample_masks)
        print(sample_images)
        
        print("Starting Image Creation")
        
        NumImages = 15
        test_generation_is_deterministic(sample_images , sample_masks , background_images, numberImages)
        print("Image Creation complete")