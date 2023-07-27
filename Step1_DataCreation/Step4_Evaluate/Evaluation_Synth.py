# import numpy as np

# # Sample performance data for four models (replace with your actual data)
# model1_performance = [0.85, 0.78, 0.89, 0.91, 0.83]
# model2_performance = [0.82, 0.79, 0.85, 0.87, 0.81]
# model3_performance = [0.88, 0.82, 0.87, 0.89, 0.90]
# model4_performance = [0.76, 0.75, 0.80, 0.79, 0.82]

# # Combine the data into a 2D array
# data = np.column_stack((model1_performance, model2_performance, model3_performance, model4_performance))

# print(data)


# Display examples with bounding box and masks
#pip install scikit-posthocs

import os
import random
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp
import detectron2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from lang_sam import LangSAM    #pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
from PIL import Image
from lang_sam.utils import draw_image


def GetYOLOResult_Contour(results):
    
    # Create a list to store dictionaries with area and count for each result
    result_data = []
    
    for result in results:
        areas = []
        image_shape = result.orig_shape
        overall_mask = np.zeros((result.orig_shape[0], result.orig_shape[1]), dtype=np.uint8)
        for j, mask in enumerate(result.masks.data):
        #   # Move the mask tensor to CPU if it's on a CUDA device
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        #   # Convert the mask to uint8 and resize it to match the image size
#            mask_np = cv2.resize((mask_np.astype( np.uint8) * 255), (result.orig_shape[1], result.orig_shape[0]))
 
            # Resize to original image size, Using INTER_NEAREST ensures resized mask remains binary with values 0 and 1, preserving its original nature. 
            mask_np_resized = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_np_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize the total area for this mask
            total_area = 0.0
    
            # Iterate through each contour and calculate its area
            for contour in contours:
                area = cv2.contourArea(contour)
                total_area += area
            
            #area_covered = np.sum(mask_np_resized)
            areas.append(total_area)
            
            # Combine the current mask with the overall mask using logical OR operation
            overall_mask = cv2.bitwise_or(overall_mask, mask_np_resized)

        result_data.append({'area': sum(areas), 'count': len(areas), 'overallMask': overall_mask})
        
    return result_data  # get count of fibres and area covered


def GetYOLOResult(results):
    
    # Create a list to store dictionaries with area and count for each result
    result_data = []
    
    for result in results:
        areas = []
        image_shape = result.orig_shape
        overall_mask = np.zeros((result.orig_shape[0], result.orig_shape[1]), dtype=np.uint8)
        for j, mask in enumerate(result.masks.data):
        #   # Move the mask tensor to CPU if it's on a CUDA device
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        #   # Convert the mask to uint8 and resize it to match the image size
#            mask_np = cv2.resize((mask_np.astype( np.uint8) * 255), (result.orig_shape[1], result.orig_shape[0]))
 
            # Resize to original image size, Using INTER_NEAREST ensures resized mask remains binary with values 0 and 1, preserving its original nature. 
            mask_np_resized = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            
            area_covered = np.sum(mask_np_resized)
            areas.append(area_covered)
            
            # Combine the current mask with the overall mask using logical OR operation
            overall_mask = cv2.bitwise_or(overall_mask, mask_np_resized)

        result_data.append({'area': sum(areas), 'count': len(areas), 'overallMask': overall_mask})
        
    return result_data  # get count of fibres and area covered



def GetDetectronResult(results):
    
    # Create a list to store dictionaries with area and count for each result
    result_data = []
    
    
        # Access the predicted instances (bounding boxes, masks, etc.)
    #instances = outputs["instances"]

    # Access the predicted masks
    predicted_masks = results["instances"].pred_masks
    predicted_masks_np = predicted_masks.cpu().numpy()
    
    image_size = results["instances"].image_size
    height, width = image_size

    overall_mask = np.zeros((height, width), dtype=np.uint8)
    areas = []
    for i, mask in enumerate(predicted_masks_np):
        
    # Calculate the contour of the mask using cv2.findContours (only external contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize the total area for this mask
        total_area = 0.0
        
        # Iterate through each contour and calculate its area
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area += area

        areas.append(total_area)

        # Combine the current mask with the overall mask using logical OR operation
        overall_mask = cv2.bitwise_or(overall_mask, mask)

    result_data.append({'area': sum(areas), 'count': len(areas), 'overallMask': overall_mask})
        
    return result_data  # get count of fibres and area covered


def GetSAMresult(results):
    
    # Create a list to store dictionaries with area and count for each result
    result_data = []

    width = results[0]['crop_box'][2]          # Cropping is the same for all the image and so represents the size of the image
    height = results[0]['crop_box'][3]
    overall_mask = np.zeros((height, width), dtype=np.uint8)
    areas = []
    for i in range(1, len(results)):   # Ignore the first finding because that always seemsto be the entire shape
         
        results[i]['segmentation']
        # Move the mask tensor to CPU if it's on a CUDA device
        mask_np = results[i]['segmentation'].detach().cpu().numpy() if isinstance(results[i]['segmentation'], torch.Tensor) else results[i]['segmentation']
        areas.append(results[i]['area'])
            
        # Combine the current mask with the overall mask using logical OR operation
        #overall_mask = cv2.bitwise_or(overall_mask, mask_np)
        overall_mask[mask_np] = 1
    result_data.append({'area': sum(areas), 'count': len(results), 'overallMask': overall_mask})
        
    return result_data  # get count of fibres and area covered

def show_anns(anns):
    # for use when displaying segment anything mask 
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def prep_mask_image(anns):
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    img = (img * 255).astype(np.uint8)  # Convert to uint8 range (0-255)
    return Image.fromarray(img)




def calculate_iou(groundTruth, predicted):
    intersection = np.logical_and(groundTruth, predicted).sum()
    union = np.logical_or(groundTruth, predicted).sum()
    iou = intersection / union
    
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(groundTruth) + np.sum(predicted))
    return iou , dice_coefficient



   # train on the GPU or on the CPU, if a GPU is not available - CPU will be very slow
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# Source Data
fibre_images = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Prepared\Test\images'
fibre_masks = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Prepared\Test\masks'
img_files = os.listdir(fibre_images) # Get list of files in the directory 

##########################################################################################
# YOLO Model
##########################################################################################
preTrainedYOLO = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\YoloLTrain1.pt'
# Load a model
YOLOmodel = YOLO(preTrainedYOLO)  # pretrained YOLOv8n model

##########################################################################################
# Detectron 
##########################################################################################
cfg = get_cfg()

#cfg.MODEL.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cfg.MODEL.DEVICE = device.type
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
###
####### Change this 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Step1_DataCreation\Step4_Evaluate\Detectron2_Trained_Model.pth'

##########################################################################################
# Segment Anything 
##########################################################################################

sam_checkpoint = r"C:\Users\dezos\Documents\Fibres\FibreAnalysis\sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = device

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_mask_generator = SamAutomaticMaskGenerator(sam)

###########################################################################################
# Language Segment anything 
###########################################################################################

samL = LangSAM()

##
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
Detect2Predictor = DefaultPredictor(cfg)

dataHolder = []
# Create a new figure
fig, axes = plt.subplots(len(img_files), 6, figsize=(15, 15))


# Set individual titles for each column
axes[0, 0].set_title("Image")
axes[0, 1].set_title("Mask")
axes[0, 2].set_title("YOLO")
axes[0, 3].set_title("MaskRCNN")
axes[0, 4].set_title("SAM")
axes[0, 5].set_title("SAMtext")


#for d in random.sample(img_files, 2):
for i, d in enumerate(random.sample(img_files, 2)):    
    fullPath = os.path.join(fibre_images, d)
    mask_path = os.path.join(fibre_masks, f"mask{d[5:]}")  # Change filename from image_ to mask _ to retrieve mask
    print(fullPath)  # Print File name 
    print(mask_path)  # Print File name 
    image = cv2.imread(fullPath)  # Load your input image   
    ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    ax = axes[i , 1]
    ax.imshow(image)
    ax.axis('off')
    ax = axes[i , 2]
    ax.imshow(ground_truth_mask)
    ax.axis('off')
    
#    results = YOLOmodel(fullPath)
    # cv2.imshow("result", image)             #https://docs.ultralytics.com/modes/predict/#plotting-results
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  

    # YOLOresult = GetYOLOResult_Contour(YOLOmodel(fullPath))   # get area and count for YOLO    
   
    yoloImage = YOLOmodel(fullPath)
    res_plotted = yoloImage[0].plot()
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)


    ax = axes[i , 2]
    ax.imshow(res_plotted_rgb)
    ax.axis('off')
    plt.show()
   
    # result_11111 = Detect2Predictor(image)
   
    # DetectronResult = GetDetectronResult(Detect2Predictor(image))

    SAMmasks = sam_mask_generator.generate(image)

    SAMresults = GetSAMresult(SAMmasks)
    
    
    # Language
    mask_image = prep_mask_image(SAMmasks)

    grayscale_image = mask_image.convert("L")

    # Create a new RGB image with grayscale values in all channels
    rgb_bw_image = Image.new("RGB", grayscale_image.size)
    rgb_bw_image.paste(grayscale_image)
    text_prompt = "line"

    SAMLmasks, SAMLboxes, SAMLlabels, SAMLlogits = samL.predict(rgb_bw_image, text_prompt, box_threshold=0.20, text_threshold=0.24)

    image_array = np.array(image)
    image101 = draw_image(image_array,  SAMLmasks,  SAMLboxes,  SAMLlabels)
    plt.figure(figsize=(20,20))
    plt.imshow(image101)

    plt.axis('off')
    plt.show()


   # Calculate IoU and Dice coefficient
#    iou_value, diceCoef= calculate_iou(ground_truth_mask, YOLOresult[-1]["overallMask"])
#    iou_value_detect, diceCoef_detect= calculate_iou(ground_truth_mask, DetectronResult[-1]["overallMask"])
    iou_value_SAM, diceCoef_SAM= calculate_iou(ground_truth_mask, SAMresults[-1]["overallMask"])    
    
   # Prepare rows for dataframe later. See this link as to why to create a list first (much faster)  https://stackoverflow.com/questions/10715965/create-a-pandas-dataframe-by-appending-one-row-at-a-time
    dataHolder.append([d, iou_value, diceCoef, iou_value_detect, diceCoef_detect])
    
    print("IoU:", iou_value)
    print("Dice Coefficient:", diceCoef)

    # YOLO 
#Display the image with full markup 
#   res_plotted = results[0].plot()  # assuming only one image per run 


plt.tight_layout()
plt.show()


#    cv2.imshow("result", res_plotted)             #https://docs.ultralytics.com/modes/predict/#plotting-results
#    # Set the window position to (x, y) coordinates (adjust these values as needed)
#    cv2.moveWindow("result", 100, 100) # window is being displayed off screen for some reason 
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()    



completeDataFrame = pd.DataFrame(dataHolder, columns=['image', 'YOLOiou','YOLODice', 'Detectiou', 'DetectDice'] )
print(completeDataFrame)


# Perform Kruskal-Wallis test
statistic, p_value = kruskal(completeDataFrame['YOLOiou'], completeDataFrame['Detectiou'])    #, data['Model 3'], data['Model 4'])

# Output the results
print("Kruskal-Wallis test statistic:", statistic)
print("P-value:", p_value)

# Interpret the results (using the same alpha as before)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the groups.")



# Perform Dunn's test with Bonferroni adjustment  - need other datagrame with all but relevant columns dropped 
dunn_results = sp.posthoc_dunn(completeDataFrame, p_adjust='bonferroni')

# Output the Dunn's test results
print("Dunn's test p-values with Bonferroni adjustment:")
print(dunn_results)

# Additional analysis (optional):
# To obtain a pairwise comparison table with significant differences marked, you can use the following:
pairwise_comparison_table = (dunn_results < alpha).astype(int)
pairwise_comparison_table.index = completeDataFrame.columns
pairwise_comparison_table.columns = completeDataFrame.columns
print("\nPairwise comparison table with significant differences (1 means significant, 0 means not significant):")
print(pairwise_comparison_table)



























## YOLO 
## Display the image with full markup 
#    res_plotted = results[0].plot()  # assuming only one image per run 
#    cv2.imshow("result", res_plotted)             #https://docs.ultralytics.com/modes/predict/#plotting-results
#    # Set the window position to (x, y) coordinates (adjust these values as needed)
#    cv2.moveWindow("result", 100, 100) # window is being displayed off screen for some reason 
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()    




    
    
    
# def  display_YOLO()   
    
#     #     # Process results list
#     for result in results:
#     #     boxes = result.boxes.data  # Boxes object for bbox outputs
#     #     masks = result.masks.data  # Masks object for segmentation masks outputs
#     #     keypoints = result.keypoints.data  # Keypoints object for pose outputs
#     #     probs = result.probs.data  # Class probabilities for classification outputs
#         image = cv2.imread(fullPath)
#         #print(masks)
        
        
#         overlay_combined = np.zeros_like(image)
        
#         for j, mask in enumerate(result.masks.data):
#         #   # Move the mask tensor to CPU if it's on a CUDA device
#             mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

#         #   # Convert the mask to uint8 and resize it to match the image size
#             mask_np = cv2.resize((mask_np.astype( np.uint8) * 255), (image.shape[1], image.shape[0]))

#                 # Create a red transparency mask (instead of grayscale)
            
#     # Convert the grayscale mask to 3-channel (RGB) format with red color
#             #red_mask = np.stack((mask_np,) * 3, axis=-1)
#             red_mask = np.zeros_like(image)
#             red_mask[:, :, 2] = mask_np  # Set the red channel to the mask value
#         #   # Create a transparency mask
#             transparency_mask = np.stack((mask_np,) * 3, axis=-1)

#         #   # Apply the mask overlay to the original image
#             # overlay = cv2.addWeighted(image, 1, transparency_mask, 0.5, 0)
#             overlay_combined = cv2.addWeighted(overlay_combined, 1, red_mask,1, 0)

#         # Combine the overlay with the original image
#         result_image = cv2.addWeighted(image, 0.7, overlay_combined, 0.3, 0)

#                 #   # Display the result
#         cv2.imshow('name',result_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()





# print('finished')    
    
    #img = cv2.imread(d["file_name"])
    #visualizer = Visualizer(img[:, :, ::-1], metadata=fibre_metadata, scale=0.5)
    #out = visualizer.draw_dataset_dict(d)
    #cv2_imshow(out.get_image()[:, :, ::-1])