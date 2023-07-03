import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re   # Regular expressions
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt




# class imagePreparation:
    
#     def __init__(self, image, mask, regions) -> None:
#        self.image = image
#        self.mask = mask
#        self.numROI = regions   # Number of regions of interest on image
       
#     def setTransformation(self): 
    
#         self.height, self.width = self.image.shape[:2]
#         self.mask = np.zeros((self.height,self.width), np.uint8)

# # Transform to gray colorspace and invert Otsu threshold the image
#         #gray = cv2.cvtColor( self.image,cv2.COLOR_BGR2GRAY)
#         gray = self.image
#        # _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# ### Perform opening (erosion followed by dilation)
#         kernel = np.ones((2,2),np.uint8)
#         # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        
#         imgCanny = cv2.Canny(gray,100,100)
#         kernel = np.ones((5,5),np.uint8)
#         self.imgEroded = cv2.dilate(imgCanny,kernel,iterations=1)

        
#     def setContours(self):    
#         contours1, hierarchy1 = cv2.findContours(self.imgEroded, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#         self.IdentifiedContours = sorted(contours1, key=cv2.contourArea, reverse=True)
#         for contour in self.IdentifiedContours:
#             area = cv2.contourArea(contour)
#             print("Contour area:", area)
            
#         print('complete')        
    
#     def cropping(self):
        
#         for i in range(self.numROI):
#             # Compute the bounding rectangle for the contour
#             x, y, w, h = cv2.boundingRect(self.IdentifiedContours[i+1])  # Largest contour 0 will be the full 
#                                                                        # image so we ignore that one
#             cropped_image = self.image[y:y+h, x:x+w]                                                           
#             cropped_mask  = self.mask[y:y+h, x:x+w]                                                                       
#         # Display the cropped image
#                     # Display the cropped image
#             plt.imshow(cropped_image)
#             plt.axis('off')  # Optional: Turn off the axes
#             plt.show()
         
 
        
        

def imagePreparation(image, mask, numROI ):
    
    height, width = image.shape[:2]
    #mask = np.zeros((height,width), np.uint8)

# Transform to gray colorspace and invert Otsu threshold the image
    #gray = cv2.cvtColor( self.image,cv2.COLOR_BGR2GRAY)
    gray = image
    # _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

### Perform opening (erosion followed by dilation)
    kernel = np.ones((2,2),np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
    imgCanny = cv2.Canny(gray,100,100)
    kernel = np.ones((5,5),np.uint8)
    imgEroded = cv2.dilate(imgCanny,kernel,iterations=1)

    
    contours1, hierarchy1 = cv2.findContours(imgEroded, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    IdentifiedContours = sorted(contours1, key=cv2.contourArea, reverse=True)
    for contour in IdentifiedContours:
        area = cv2.contourArea(contour)
        print("Contour area:", area)
        
    print('complete')        


    cropped_images = []  # List to store cropped images
    cropped_masks = []   # List to store cropped masks
    
    for i in range(numROI):
        # Compute the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(IdentifiedContours[i+1])  # Largest contour 0 will be the full 
                                                                    # image so we ignore that one
        cropped_images.append(image[y:y+h, x:x+w])  # Add cropped image to the list
        cropped_masks.append(mask[y:y+h, x:x+w])    # Add cropped mask to the list
    
    return cropped_images, cropped_masks
                                                                                   
    # Display the cropped image
                # Display the cropped image
        # plt.imshow(cropped_image)
        # plt.axis('off')  # Optional: Turn off the axes
        # plt.show()
        
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
    
    
    
current_directory = os.getcwd()
print('Current Directory', current_directory)

rawImages = os.path.join(current_directory, 'Data', 'synth', 'images', '')
rawMasks = os.path.join(current_directory, 'Data', 'synth', 'masks', '')

preparedImages = os.path.join(current_directory, 'Data', 'Prepared', 'images', '')
preparedMasks = os.path.join(current_directory, 'Data', 'Prepared', 'masks', '')
preparedLabels = os.path.join(current_directory, 'Data', 'Prepared', 'labels', '')


tempImageFilenames = os.listdir(rawImages)
imageFilenames = [item for item in tempImageFilenames if os.path.isfile(os.path.join(rawImages, item))]


# pattern = r"\b(" + "|".join(one_box_files) + r")\b"

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
        
    croppedImgs, croppedMasks = imagePreparation(img, mask, 2)
    
    counter = 0
    for croppedImg, croppedMask in zip(croppedImgs, croppedMasks):
       counter += 1
       
       imageName_without_extension, imageExtension = os.path.splitext(filename)
       maskname_without_extension, maskExtension = os.path.splitext(maskFilename)
      
       targetImageFile = os.path.join(preparedImages, f"{imageName_without_extension}_{counter}{imageExtension}")
       targetMaskFile = os.path.join(preparedMasks, f"{maskname_without_extension}_{counter}{maskExtension}")
       
       cv2.imwrite(targetImageFile, croppedImg)  # Save the image to file
       cv2.imwrite(targetMaskFile, croppedMask)  # Save the mask to file
       store_polygons(preparedLabels, f"{labelFilename}_{counter}.txt",  get_contours(croppedMask)) 


