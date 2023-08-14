# from https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/


# Ultra lytics support SAM for inference only https://docs.ultralytics.com/models/sam/#available-models-and-supported-tasks


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch.nn.functional as F
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import torch
from matplotlib.patches import Rectangle
from datetime import datetime
from collections import defaultdict

import torch

from segment_anything.utils.transforms import ResizeLongestSide


current_directory = os.getcwd()
OutPreparedImages = os.path.join(current_directory, 'Data', 'Prepared', 'Train', 'images', '')
OutPreparedMasks = os.path.join(current_directory, 'Data', 'Prepared', 'Train', 'masks', '')

# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb

#########################################################################################
# Temp 
#########################################################################################
def show_anns(anns):
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


model_type = 'vit_b'
#checkpoint = 'sam_vit_b_01ec64.pth'
checkpoint = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Training\segmentAnything\sam_vit_b_01ec64.pth'
#Training\segmentAnything\sam_vit_b_01ec64.pth
   # train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
#sam_model.train()

image = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Prepared\Train\images\image_2023-07-14_21-27-23-239174_1.png'
image1 = cv2.imread(image) 
mask_generator = SamAutomaticMaskGenerator(sam_model)
masks = mask_generator.generate(image1)




plt.figure(figsize=(20,20))
plt.imshow(image1)
show_anns(masks)
plt.axis('off')
plt.show() 




# Set up predictors for both tuned and original models
#from segment_anything import sam_model_registry, SamPredictor
#predictor_original = SamAutomaticMaskGenerator(sam_model_orig)


#########################################################################################

#########################################################################################



def show_masks(masks, ax, random_color=False):
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_boxes(bboxes, ax):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)


bbox_coords = {}
ground_truth_masks = {}
for f in sorted(Path(OutPreparedMasks).iterdir())[:100]:
    k = f.stem[:]
    
    mask = cv2.imread(f.as_posix(), cv2.IMREAD_GRAYSCALE)

    _, mask1 = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask1.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
       # height, width, _ = im.shape
        height, width = mask.shape
        bounding_boxes.append(np.array([x, y, x + w, y + h]))
    if len(bounding_boxes) > 0:
        bbox_coords[k] = bounding_boxes

    ground_truth_masks[k] = [~(mask == 0)] * len(bbox_coords[k])

    image_path = os.path.join(OutPreparedImages, f'image{k[4:]}.png')    
    image = cv2.imread(image_path)


model_type = 'vit_b'
#checkpoint = 'sam_vit_b_01ec64.pth'
checkpoint = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Training\segmentAnything\sam_vit_b_01ec64.pth'
#Training\segmentAnything\sam_vit_b_01ec64.pth
   # train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()

############################################################################################
# Preprocess the images
############################################################################################


transformed_data = defaultdict(dict)
for k in bbox_coords.keys():

  image_path = os.path.join(OutPreparedImages, f'image{k[4:]}.png')    
  image = cv2.imread(image_path) 
  #image = cv2.imread(f'{OutPreparedImages}{k}.png')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
  transform = ResizeLongestSide(sam_model.image_encoder.img_size)
  input_image = transform.apply_image(image)
  input_image_torch = torch.as_tensor(input_image, device=device)
  transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  
  
  input_image = sam_model.preprocess(transformed_image)
  original_image_size = image.shape[:2]
  input_size = tuple(transformed_image.shape[-2:])

  transformed_data[k]['image'] = input_image
  transformed_data[k]['input_size'] = input_size
  transformed_data[k]['original_image_size'] = original_image_size

############################################################################################

############################################################################################

# Set up the optimizer, hyperparameter tuning will improve performance here
#lr = 1e-4
lr = 1e-2
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

#loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.BCELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()


keys = list(set(bbox_coords.keys()))   # Get unique list of keys 
############################################################################################

############################################################################################

num_epochs = 10
losses = []

for epoch in range(num_epochs):
  epoch_losses = []
  
  for k in keys:
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']
    
    # No grad here as we don't want to optimize the encoders
    with torch.no_grad():
      image_embedding = sam_model.image_encoder(input_image)
      
      prompt_boxes = bbox_coords[k]  # Multiple bounding boxes
      prompt_boxes1 = np.array(prompt_boxes)  # Convert to NumPy array
      boxes = transform.apply_boxes(prompt_boxes1, original_image_size)
      
      boxes_torch = torch.as_tensor(boxes, dtype=torch.float, device=device)
      
      sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
          points=None,
          boxes=boxes_torch,
          masks=None,
      )
      
    low_res_masks, iou_predictions = sam_model.mask_decoder(
      image_embeddings=image_embedding,
      image_pe=sam_model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=True,  # Output multiple masks
    )
    
    
    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    # Convert RGB image to grayscale
    gray_image = torch.mean(upscaled_masks, dim=1, keepdim=True)
    binary_masks = normalize(threshold(gray_image, 0.0, 0))
    
    gt_masks_resized = []
    for gt_mask in ground_truth_masks[k]:  # Loop over multiple ground truth masks
        gt_mask_resized = torch.from_numpy(np.resize(gt_mask, (1, gt_mask.shape[0], gt_mask.shape[1]))).to(device)        
        
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
        gt_masks_resized.append(gt_binary_mask)

# Stack the individual gt_binary_mask tensors along the batch dimension
    gt_masks_tensor = torch.stack(gt_masks_resized, dim=0)

    
 #   loss = loss_fn(binary_masks, torch.stack(gt_masks_resized))
    loss = loss_fn(binary_masks, gt_masks_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_losses.append(loss.item())
  
  losses.append(epoch_losses)
  print(f'EPOCH: {epoch}')
  print(f'Mean loss: {mean(epoch_losses)}')
  
  # Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set the filename with the concatenated timestamp
filename = f"SegmentAnythingModel_{timestamp}.pt"
  
torch.save(sam_model.state_dict(), filename)
  
mean_losses = [mean(x) for x in losses]
mean_losses

plt.plot(list(range(len(mean_losses))), mean_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.show()

