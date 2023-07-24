import os

# Clean up Directories by removing all existing raw and prepared files.

def delete_files_in_directory(directory):
    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    if len(file_list) > 0: 
        # Iterate over the files and delete them
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        print("Files deleted from :" , file_list)

current_directory = os.getcwd()

# Call the function to delete files in the directory
for stageDirectory in ["Train", "Val", "Test"]: 
    InRawImages = os.path.join(current_directory, 'Data', 'synth',stageDirectory, 'images', '')
    InRawMasks = os.path.join(current_directory, 'Data', 'synth', stageDirectory,'masks', '')
    OutPreparedImages = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'images', '')
    OutPreparedMasks = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'masks', '')
    OutPreparedLabels = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'labels', '')

    OutYOLOPreparedImages = os.path.join(current_directory, 'Data', 'Prepared', 'YOLO', 'images',stageDirectory, '')
    OutYOLOPreparedLabels = os.path.join(current_directory, 'Data', 'Prepared', 'YOLO', 'labels', stageDirectory, '')


    delete_files_in_directory(InRawImages)
    delete_files_in_directory(InRawMasks)
    delete_files_in_directory(OutPreparedImages)
    delete_files_in_directory(OutPreparedMasks)
    delete_files_in_directory(OutPreparedLabels)
    delete_files_in_directory(OutYOLOPreparedImages)    
    delete_files_in_directory(OutYOLOPreparedLabels)    

print("Files have been deleted")