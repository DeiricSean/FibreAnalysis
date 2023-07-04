import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re   # Regular expressions
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imutils       


def imagePreparation(image, mask, numROI ):
    
    height, width = image.shape[:2]
    imageArea = height * width
      
    resized = imutils.resize(image, width=300)
    
    ratio = image.shape[0] / float(resized.shape[0])

# # Apply Gaussian blur
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# # Morphological operations 

    # Perform morphological operations to connect and thicken the lines
    # kernel = np.ones((3, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Smooth the lines using a median filter
 #   smoothed = cv2.medianBlur(closed, 3)


    dilated_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # MORPH_CLOSE is useful for closing small gaps
    # Canny edge detection
    edges = cv2.Canny(dilated_edges, 100, 200)

#######################################################################################################################
    # Remove lines from the image
    #result = cv2.bitwise_and(image, cv2.bitwise_not(mask))

    contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    IdentifiedContours = sorted(contours1, key=cv2.contourArea, reverse=True)
    
    if len(IdentifiedContours) >= 2:
    # Get the areas of the first two contours
    # This is to identify whether the image has one or two main grid squares
        if abs(cv2.contourArea(IdentifiedContours[0]) / cv2.contourArea(IdentifiedContours[1])) < 2 :
            numROI = 2
        else:
            numROI = 1

    for x in range(numROI):
        
    #for contour in IdentifiedContours:
        IdentifiedContours[x][:, 0, 0] = (IdentifiedContours[x][:, 0, 0] * ratio).astype(int)  # Scale the x-coordinates
        IdentifiedContours[x][:, 0, 1] = (IdentifiedContours[x][:, 0, 1] * ratio).astype(int)  # Scale the y-coordinates

            
    print('image area', imageArea, abs(cv2.contourArea(IdentifiedContours[0])) / imageArea , 
                        abs(cv2.contourArea(IdentifiedContours[1]))/ imageArea,
                        abs(cv2.contourArea(IdentifiedContours[2]))/ imageArea )            
    
    print('areas', abs(cv2.contourArea(IdentifiedContours[0])), abs(cv2.contourArea(IdentifiedContours[1])),
          abs(cv2.contourArea(IdentifiedContours[2])))
    
    
    cropped_images = []  # List to store cropped images
    cropped_masks = []   # List to store cropped masks
    
    for i in range(numROI):
        # Compute the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(IdentifiedContours[i])  # Largest contour 0 will be the full 
                                                                    # image so we ignore that one
        cropped_images.append(image[y:y+h, x:x+w])  # Add cropped image to the list
        cropped_masks.append(mask[y:y+h, x:x+w])    # Add cropped mask to the list
    
        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)
        #ax.imshow(resized)
        
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

