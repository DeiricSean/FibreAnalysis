import cv2
import numpy as np
import matplotlib.pyplot as plt

#####################################################



#### have a look at adaptive thresholding 


###########################################


class imagePreparation:
    
    def __init__(self, image) -> None:
       self.image = image

       
    def setTransformation(self): 
    
        self.height, self.width = self.image.shape[:2]
        self.mask = np.zeros((self.height,self.width), np.uint8)

# Transform to gray colorspace and invert Otsu threshold the image
        gray = cv2.cvtColor( self.image,cv2.COLOR_BGR2GRAY)
       # _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

### Perform opening (erosion followed by dilation)
        kernel = np.ones((2,2),np.uint8)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        
        imgCanny = cv2.Canny(gray,100,100)
        kernel = np.ones((5,5),np.uint8)
        self.imgEroded = cv2.dilate(imgCanny,kernel,iterations=1)

        
    def setContours(self):    
        contours1, hierarchy1 = cv2.findContours(self.imgEroded, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        self.IdentifiedContours = sorted(contours1, key=cv2.contourArea, reverse=True)
        for contour in self.IdentifiedContours:
            area = cv2.contourArea(contour)
            print("Contour area:", area)
    
    