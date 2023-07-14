
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from PIL import Image
import cv2  
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision.transforms.functional as TF
import random
import imutils    
from torch.utils.data._utils.collate import default_collate
from torchvision.models.detection import maskrcnn_resnet50_fpn , MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt 

# set the input height and width
#INPUT_HEIGHT = 1064
#INPUT_WIDTH = 1064


# initialize our data augmentation functions
ChangeType = transforms.ConvertImageDtype(torch.float32)


trainTransforms = transforms.Compose([transforms.ToTensor()])
valTransforms = transforms.Compose([transforms.ToTensor()])



def get_transform(train):
    transforms = []
    transforms.append(TF.pil_to_tensor())
    transforms.append(TF.convert_image_dtype(torch.float))

    return transforms.Compose(transforms)


def my_segmentation_transforms(image, segmentation):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    # more transforms ...
    return image, segmentation

def collate_fn(batch):
    return tuple(zip(*batch))


transform = transforms.Compose(
    [transforms.PILToTensor(),
     transforms.ConvertImageDtype(torch.float)
    ])

class FibreDataset(torch.utils.data.Dataset):
    def __init__(self, root, stage, transforms):
        self.root = root
        self.transforms = transforms
        self.stage = stage
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'Prepared', self.stage, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'Prepared', self.stage, 'masks'))))

    def __getitem__(self, idx):
        # load images and masks
        
        img_path = os.path.join(self.root, 'Prepared', self.stage, 'images', self.imgs[idx])
        mask_path = os.path.join(self.root, 'Prepared', self.stage, 'masks', self.masks[idx])
#        print(self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # convert the Image into a numpy array
        mask = np.array(mask)
        # Create an empty list to store the binary masks
        masks = []
        
        # Get the contours of the image - to find the grid polygons
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        num_objs = len(contours)
        # Create an empty numpy array to store the masks
        mask_shape = mask.shape
        masks = np.zeros((num_objs,) + mask_shape, dtype=np.uint8)

        # Iterate over each contour and update the masks array
        for i, contour in enumerate(contours):
                # Check if the contour area is greater than 0
            if cv2.contourArea(contour) == 0:
                contour[-1][-1] += 1   # A straight line is being picked up as having no area 
                                       # so this is a crude way of ensuring there are no 
                                       # zero value areas.
               
            # Create a new binary mask for the current contour
            contour_mask = np.zeros_like(mask, dtype=np.uint8)
            # Draw the contour on the binary mask
            cv2.drawContours(contour_mask, [contour], 0, (255), thickness=cv2.FILLED)
            # Update the masks array with the contour mask
            masks[i] = contour_mask

        boxes = [] 
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(contours[i])
            boxes.append([x, y, x+w, y+h])
            
        # convert everything into a torch.Tensor
        # there is only one class
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(torch.ones((num_objs,), dtype=torch.int64))
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)    

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        images1 = list(image.to(device) for image in inputs)
        targets1 = [{k: v.to(device) for k, v in t.items()} for t in labels]

       # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss_dict = model(images1, targets1)

        losses = sum(loss for loss in loss_dict.values())
        print(i,'loss:', losses.item())
        
        losses.backward()
        optimizer.step()

        # Gather data and report
        running_loss += losses.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == "__main__":
   

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
    model.to(device)# move model to the right device

    #loss_fn = torch.nn.CrossEntropyLoss()  # might be for multiclass classification
    current_directory = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data'

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    dataset = FibreDataset(current_directory, 'Train',  trainTransforms)    
    dataset_test = FibreDataset(current_directory, 'Test', valTransforms)
    

    print(len(dataset))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn)

    num_classes = 2

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)  # https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
    # Optimizers specified in the torch.optim package 
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 1  # Set number of epochs
    loss_fn = torch.nn.BCEWithLogitsLoss
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader_test):
                vinputs, vlabels = vdata
                images1 = list(image.to(device) for image in vinputs)
                targets1 = [{k: v.to(device) for k, v in t.items()} for t in vlabels]            
                voutputs = model(images1,targets1)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                
                
                # inputs, labels = vdata
                # images1 = list(image.to(device) for image in inputs)
                # targets1 = [{k: v.to(device) for k, v in t.items()} for t in labels]
                # voutputs = model(images1, )

        # # Zero your gradients for every batch!
        # optimizer.zero_grad()

        # loss_dict = model(images1, targets1)
                
                

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
    
    