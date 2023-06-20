
from generate_test import test_generation_is_deterministic

if __name__ == '__main__':
    
    import os

    current_directory = os.getcwd()
    print(current_directory)
    #current_directory = os.path.dirname(os.path.abspath(__file__))
    sample_images = os.path.join(current_directory, 'Data', 'Yolo', 'images', '')
    sample_masks = os.path.join(current_directory, 'Data', 'Yolo', 'masks', '')
    print(sample_masks)
    print(sample_images)
    
    print("is this working?")
    test_generation_is_deterministic(sample_images , sample_masks )
    print("perhaps")