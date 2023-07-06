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
      
    resized = imutils.resize(image, width=300)  # The height is also adjusted to preserve the aspect ratio
    
    ratio = image.shape[0] / float(resized.shape[0])
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Morphological operations to connect and thicken the lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # MORPH_CLOSE is useful for closing small gaps
    # Canny edge detection
    edges = cv2.Canny(dilated_edges, 100, 200)

    # Get the contours of the image - to find the grid polygons
    contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    IdentifiedContours = sorted(contours1, key=cv2.contourArea, reverse=True)
    
    if len(IdentifiedContours) >= 2:
    # Get the areas of the first two contours,this is to identify whether the image has one or two main grid squares
        if abs(cv2.contourArea(IdentifiedContours[0]) / cv2.contourArea(IdentifiedContours[1])) < 2 :
            numROI = 2
        else:
            numROI = 1

    # Rescale the image coordinates from the smaller resized image to the actual image 
    for x in range(numROI):
    #for contour in IdentifiedContours:
        IdentifiedContours[x][:, 0, 0] = (IdentifiedContours[x][:, 0, 0] * ratio).astype(int)  # Scale the x-coordinates
        IdentifiedContours[x][:, 0, 1] = (IdentifiedContours[x][:, 0, 1] * ratio).astype(int)  # Scale the y-coordinates
    
    cropped_images = []  # List to store cropped images
    cropped_masks = []   # List to store cropped masks
    
    for i in range(numROI):
        # Compute the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(IdentifiedContours[i])  # Largest contour 0 will be the full 
                                                                    # image so we ignore that one
        cropped_images.append(image[y:y+h, x:x+w])  # Add cropped image to the list
        cropped_masks.append(mask[y:y+h, x:x+w])    # Add cropped mask to the list
    
    return cropped_images, cropped_masks
                                                                                   

# Convert masks into YOLO suitable labels - based on the masks rather than the images
# get_contours and store_polygons are from https://github.com/computervisioneng/image-segmentation-yolov8
#  
def get_contours( inboundMask ):
 
    _, mask = cv2.threshold(inboundMask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(inboundMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
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
    

# Main processing 
def processSynthImages(rawImages, rawMasks, preparedImages, preparedMasks, preparedLabels):

    # Get the list of files to process
    tempImageFilenames = os.listdir(rawImages)
    imageFilenames = [item for item in tempImageFilenames if os.path.isfile(os.path.join(rawImages, item))]
    
    for filename in imageFilenames:
        # Prepare tarket filenames and locations for mask and image
        ImageFile = os.path.join(rawImages, filename)
        maskFilename = filename.replace('image', 'mask')
        labelFilename = filename.replace('image', 'label')
        
        MaskFile = os.path.join(rawMasks, maskFilename)

        # read image
        img = cv2.imread(ImageFile, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(MaskFile, cv2.IMREAD_GRAYSCALE)
        
        # Crop image to show only the grid square/rectangles            
        croppedImgs, croppedMasks = imagePreparation(img, mask, 3)
        
        # Save the resulting images, masks and labels to the target directories        
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
print('Processing Images')

for stageDirectory in ["Train", "Val", "Test"]: 

    InRawImages = os.path.join(current_directory, 'Data', 'synth',stageDirectory, 'images', '')
    InRawMasks = os.path.join(current_directory, 'Data', 'synth', stageDirectory,'masks', '')

    OutPreparedImages = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'images', '')
    OutPreparedMasks = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'masks', '')
    OutPreparedLabels = os.path.join(current_directory, 'Data', 'Prepared', stageDirectory, 'labels', '')

    processSynthImages(InRawImages, InRawMasks, OutPreparedImages, OutPreparedMasks, OutPreparedLabels)

print('Processing Complete')
