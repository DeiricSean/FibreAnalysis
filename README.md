# FibreAnalysis

There are four sections to this project

## Section 1 - Data Creation

This covers the cleaning up of any old image files left in directories from previous runs. Then it generates a new set of images for training, validation and testing. There are options to run this locally or in google colab.

## Section 2 - Data Preparation

This covers taking the newly created images from step 1, cropping out the areas of interest from the image and related mask, placing them in a new directory and also generating the YOLO text file which contains the polygon dimensions of the fibres.

## Section 3 - Training 

Train and validate YOLO, Mask RCNN and Segment Anything

## Section 4 - Evaluation 

Evaluate the results of tests on YOLO, Mask RCNN, Segment Anything and Language Segment-Anything. Compare these using tests such as Kruskal-Wallis and Dunn. Display the results graphically using heatmaps, box plots etc.  