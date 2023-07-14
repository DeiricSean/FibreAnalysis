import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn , MaskRCNN_ResNet50_FPN_Weights
batchSize=10
imageSize=[600,600]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
#trainDir="/home/breakeroftime/Documents/Datasets/LabPics/LabPicsChemistry/Train"

current_directory = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data'

trainMaskDirectory1 = os.path.join(current_directory, 'Prepared', 'Train', 'masks')
maskDir1 = list(sorted(os.listdir(trainMaskDirectory1)))

trainImageDirectory = os.path.join(current_directory, 'Prepared', 'Train', 'images')

trainDir = list(sorted(os.listdir(trainImageDirectory)))
#trainDir = list(sorted(os.listdir(os.path.join(current_directory, 'Prepared', 'Train', 'images'))))



def loadData():
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    for i in range(batchSize):
        idx=random.randint(0,len(trainDir)-1)
        #img = cv2.imread(os.path.join(imgs[idx], "Image.jpg"))
                
        img = cv2.imread(os.path.join(trainImageDirectory, trainDir[idx]))
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        
        
        fullMask = (cv2.imread(os.path.join(trainMaskDirectory1, maskDir1[idx]), 0) > 0).astype(np.uint8)  # Read vesse instance mask
        fullMask=cv2.resize(fullMask,imageSize,cv2.INTER_NEAREST)
        
        #########################################################################
        
        
        
             # Get the contours of the image - to find the grid polygons
        contours, _ = cv2.findContours(fullMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        num_objs = len(contours)
        # Create an empty numpy array to store the masks
        #mask_shape = fullMask.shape
        masks=[]
        #masks = np.zeros((num_objs,) + mask_shape, dtype=np.uint8)

        # Iterate over each contour and update the masks array
        for i, contour in enumerate(contours):
                # Check if the contour area is greater than 0
            if cv2.contourArea(contour) >  0:
            #     # contour[-1][-1] += 2  # A straight line is being picked up as having no area 
            #     #                        # so this is a crude way of ensuring there are no 
            #     #                        # zero value areas.
               
            # # Create a new binary mask for the current contour
            #     contour_mask = np.zeros_like(fullMask, dtype=np.uint8)
            #     # Draw the contour on the binary mask
            #     cv2.drawContours(contour_mask, [contour], 0, (255), thickness=cv2.FILLED)
            #     # Update the masks array with the contour mask
            #     masks[i] = contour_mask
        
        
                # Create a new binary mask for the current contour
                contour_mask = np.zeros_like(fullMask, dtype=np.uint8)
                # Draw the contour on the binary mask
                cv2.drawContours(contour_mask, [contour], 0, (255), thickness=cv2.FILLED)
                # Append the contour mask to the masks list
                masks.append(contour_mask)

        # Convert the masks list to a NumPy array
        masks = np.array(masks)
        
        
        
        #########################################################################
        
        # masks=[]
        # for mskName in os.listdir(maskDir):
        #     vesMask = (cv2.imread(maskDir+'/'+mskName, 0) > 0).astype(np.uint8)  # Read vesse instance mask
        #     vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST)
        #     masks.append(vesMask)# get bounding box coordinates for each mask
            
        num_objs = len(masks)
        if num_objs==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

for i in range(10001):
            images, targets = loadData()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            print(i,'loss:', losses.item())
            if i%500==0:
                torch.save(model.state_dict(), str(i)+".torch")
