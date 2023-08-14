#    Author  : Derek O'Sullivan
#    Date    : 14/08/23

#    Purpose : This creates a set of synth images on local PC for the train, validation and test folders

# Adapted from https://github.com/hughfdjackson/fluorescent-fibre-counting.git



import os
from generate_test import test_generation_is_deterministic

if __name__ == '__main__':

    current_directory = os.getcwd()
    print('Current Directory', current_directory)

    # Create defined number of synthetic images
    for stageDirectory, numberImages in [("Train", 1000), ("Val", 500), ("Test", 200)]:

        background_images = os.path.join(current_directory, 'Data', 'Backgrounds', '')
        sample_images = os.path.join(current_directory, 'Data', 'synth', stageDirectory,'images', '')
        sample_masks = os.path.join(current_directory, 'Data', 'synth',  stageDirectory, 'masks', '')
        #sample_labels = os.path.join(current_directory, 'Data', 'synth',  stageDirectory, 'labels', '') # labels done in secondary processing
        
        print("Starting Image Creation")
        
        NumImages = 15
        test_generation_is_deterministic(sample_images , sample_masks , background_images, numberImages)
        print("Image Creation complete")