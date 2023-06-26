#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=LsuCjbUoI7A

"""
@author: Sreenivas Bhattiprolu
"""

 
import numpy as np
import cv2
import pandas as pd
import random

 
def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
########################################
#Geerate OTHER FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #Feature 11 is Variance with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe


    return df


#########################################################

#Applying trained model to segment multiple files. 

import glob
import pickle
from matplotlib import pyplot as plt
import os

imagePath = r"C:\Users\dezos\Documents\Fibres\FibreAnalysis"

sample_images = os.path.join(imagePath, 'Data', 'synth', 'images', 'test', '')
sample_masks = os.path.join(imagePath, 'Data', 'synth', 'masks', '')

#file_pattern = os.path.join(sample_images, "*.png")
file_pattern = os.path.join(sample_images, "*.jpg")

# Use glob to retrieve all matching file paths
file_paths = glob.glob(file_pattern)
# Extract filenames without the path
imgs = [os.path.basename(file_path) for file_path in file_paths]




filename = "RandomForest_Segmentation_model"
loaded_model = pickle.load(open(filename, 'rb'))



for i in range(1):
    idx=random.randint(0,len(imgs)-1)
    print(os.path.join(sample_images , imgs[idx]))
    img1 = cv2.imread(os.path.join(sample_images , imgs[idx]))
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 


#Call the feature extraction function.
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    
   # name = file.split("e_")
    plt.imsave('segResult.png', segmented, cmap ='jet')

#Above, we are splitting the file path into 2 -> creates a list with 2 entries
#Then we are taking the second half of name to save segmented images with that name

