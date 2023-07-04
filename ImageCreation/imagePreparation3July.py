import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re   # Regular expressions
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
       
        

def imagePreparation(image, mask, numROI ):
    
    height, width = image.shape[:2]
    imageArea = height * width

# # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
#  #   equalized = cv2.equalizeHist(image)

#     # Convert the image to float32 data type
#  #   image_float = image.astype(np.float32)

#     # Calculate the minimum and maximum pixel values
#  #   min_val = np.min(image_float)
#  #   max_val = np.max(image_float)

#     # Normalize the image between 0 and 1
#   #  normalized = (image_float - min_val) / (max_val - min_val)

# # Display the image using plt
#   #  plt.imshow(normalized)
#   #  plt.axis('off')  # Turn off axis ticks and labels
#   #  plt.show()
    
    _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Apply adaptive thresholding to create a binary image
    #thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=2)

#     # Canny edge detection
# # Morphological operations (dilation)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(thresh, kernel, iterations=1)
    #dilated_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #dilated_edges = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)


    edges = cv2.Canny(dilated_edges, 100, 200)






    # Perform morphological operations to connect and thicken the lines
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Smooth the lines using a median filter
 #   smoothed = cv2.medianBlur(closed, 3)

    # Invert the image to get the filled lines
    # filled_lines = cv2.bitwise_not(smoothed)
    # plt.imshow(filled_lines)
    # plt.axis('off')  # Turn off axis ticks and labels
    # plt.show()


    
    # Apply Gaussian smoothing
    # blurred = cv2.GaussianBlur(dilated_edges, (5, 5), 0)
   


#######################################################################################################################
    # Remove lines from the image
    #result = cv2.bitwise_and(image, cv2.bitwise_not(mask))

    contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# Filter contours based on area
    #area_threshold = 670000
    #filtered_contours = [contour for contour in contours1 if cv2.contourArea(contour) < area_threshold]


    IdentifiedContours = sorted(contours1, key=cv2.contourArea, reverse=True)
    
    if len(IdentifiedContours) >= 2:
    # Get the areas of the first two contours
    # The contour at point 0 is usually either the entire image or a subsection larger than the region of interest
        if abs(cv2.contourArea(IdentifiedContours[0]) / cv2.contourArea(IdentifiedContours[1])) < 2 :
            numROI = 2
        else:
            numROI = 1

    #shape = dilated_edges.shape
    #imageArea1 = height1 * width1
            
    print('image area', imageArea, abs(cv2.contourArea(IdentifiedContours[0])) / imageArea , 
                        abs(cv2.contourArea(IdentifiedContours[1]))/ imageArea,
                        abs(cv2.contourArea(IdentifiedContours[2]))/ imageArea )            
    
    print('areas', abs(cv2.contourArea(IdentifiedContours[0])), abs(cv2.contourArea(IdentifiedContours[1])),
          abs(cv2.contourArea(IdentifiedContours[2])))
    
    
    cropped_images = []  # List to store cropped images
    cropped_masks = []   # List to store cropped masks
    
    for i in range(numROI+1):
        # Compute the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(IdentifiedContours[i])  # Largest contour 0 will be the full 
                                                                    # image so we ignore that one
        cropped_images.append(image[y:y+h, x:x+w])  # Add cropped image to the list
        cropped_masks.append(mask[y:y+h, x:x+w])    # Add cropped mask to the list
    
        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='blue')

        # Add the patch to the axes
        ax.add_patch(rect)

        # Show the image with the bounding box
        plt.show()
    
    
    
    return cropped_images, cropped_masks
                                                                                   

# get_contours and store_polygons taken from  https://github.com/computervisioneng/image-segmentation-yolov8
# used to convert masks into YOLO suitable labels
def get_contours( inboundMask ):

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
    

def processSynthImages(rawImages, rawMasks, preparedImages, preparedMasks, preparedLabels):

    tempImageFilenames = os.listdir(rawImages)
    imageFilenames = [item for item in tempImageFilenames if os.path.isfile(os.path.join(rawImages, item))]

    # Print the filenames
    for filename in imageFilenames:
        ImageFile = os.path.join(rawImages, filename)
        maskFilename = filename.replace('image', 'mask')
        labelFilename = filename.replace('image', 'label')
        
        MaskFile = os.path.join(rawMasks, maskFilename)
        
        
            # read image
        img = cv2.imread(ImageFile, cv2.IMREAD_UNCHANGED)
        #mask = cv2.imread(MaskFile, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(MaskFile, cv2.IMREAD_GRAYSCALE)
            
        croppedImgs, croppedMasks = imagePreparation(img, mask, 3)
        
        counter = 0
        for croppedImg, croppedMask in zip(croppedImgs, croppedMasks):
            counter += 1
            
            imageName_without_extension, imageExtension = os.path.splitext(filename)
            maskname_without_extension, maskExtension = os.path.splitext(maskFilename)
            labelname_without_extension, _ = os.path.splitext(labelFilename)
            
            targetImageFile = os.path.join(preparedImages, f"{imageName_without_extension}_{counter}{imageExtension}")
            targetMaskFile = os.path.join(preparedMasks, f"{maskname_without_extension}_{counter}{maskExtension}")
            

            
            cv2.imwrite(targetImageFile, croppedImg)  # Save the image to file
            cv2.imwrite(targetMaskFile, croppedMask)  # Save the mask to file
            store_polygons(preparedLabels, f"{labelname_without_extension}_{counter}.txt",  get_contours(croppedMask)) 



current_directory = os.getcwd()
print('Current Directory', current_directory)

for stageDirectory in ["Train", "Val", "Test"]: 

    InRawImages = os.path.join(current_directory, 'Data', 'synth',stageDirectory, 'images', '')
    InRawMasks = os.path.join(current_directory, 'Data', 'synth', stageDirectory,'masks', '')

    OutPreparedImages = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'images', '')
    OutPreparedMasks = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'masks', '')
    OutPreparedLabels = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'labels', '')

    processSynthImages(InRawImages, InRawMasks, OutPreparedImages, OutPreparedMasks, OutPreparedLabels)

