import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
import glob


batchSize=2
imageSize=[600,600]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
trainDir="/home/breakeroftime/Documents/Datasets/LabPics/LabPicsChemistry/Train"

imgs=[]

# Update in colab
imagePath = r"C:\Users\dezos\Documents\Fibres\FibreAnalysis"

sample_images = os.path.join(imagePath, 'Data', 'synth', 'images', '')
sample_masks = os.path.join(imagePath, 'Data', 'synth', 'masks', '')


imgs=[]
# for pth in os.listdir(sample_images):
#     imgs.append(pth )

# Define the file pattern
file_pattern = os.path.join(sample_images, "*.png") 

# Use glob to retrieve all matching file paths
file_paths = glob.glob(file_pattern)
# Extract filenames without the path
imgs = [os.path.basename(file_path) for file_path in file_paths]
    
    
    
import random
batchSize=2
imageSize=[600,600]

def loadData():
  batch_Imgs=[]
  batch_Data=[]
  for i in range(batchSize):        
        idx=random.randint(0,len(imgs)-1)
        img = cv2.imread(os.path.join(sample_images , imgs[idx]))
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)       
        
        sample_masks = os.path.join(imagePath, 'Data', 'synth', 'masks', '')
        masks=[]

        filename, extension = os.path.splitext(imgs[idx])
        maskFile =  "mask" + filename[5:] + extension

        vesMask = cv2.imread(os.path.join(sample_masks , maskFile), 0)     


        vesMask = (vesMask > 0).astype(np.uint8) 
        vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST)
        
        masks.append(vesMask)        
        num_objs = len(masks)
        
        if num_objs==0: return loadData()        
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        tempImg = img
        
        img = torch.as_tensor(img, dtype=torch.float32)        
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   
        data["masks"] = masks        
        batch_Imgs.append(img)
        batch_Data.append(data)
        
        # Draw the bounding box on the image
        cv2.rectangle(tempImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Display the image
        cv2.imshow("Image with Bounding Box", tempImg)

        # # Wait for a key press and close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
  
  batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
  batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
  return batch_Imgs, batch_Data


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
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