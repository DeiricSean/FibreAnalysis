# from https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch.nn.functional as F
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
import torch
from matplotlib.patches import Rectangle


current_directory = os.getcwd()
OutPreparedImages = os.path.join(current_directory, 'Data', 'Prepared', 'Train', 'images', '')
OutPreparedMasks = os.path.join(current_directory, 'Data', 'Prepared', 'Train', 'masks', '')

# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb


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
  #if k not in stamps_to_exclude:
    #im = cv2.imread(f.as_posix())
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.imread(f.as_posix(), cv2.IMREAD_GRAYSCALE)

    _, mask1 = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask1.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    #contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
       # height, width, _ = im.shape
        height, width = mask.shape
        bounding_boxes.append(np.array([x, y, x + w, y + h]))
    if len(bounding_boxes) > 0:
        bbox_coords[k] = bounding_boxes


    #for k in bbox_coords.keys():
    # gt_grayscale = cv2.imread(f'ground-truth-pixel/ground-truth-pixel/{k}-px.png', cv2.IMREAD_GRAYSCALE)
    #ground_truth_masks[k] = [(gray == 0)] * len(bbox_coords[k])
    ground_truth_masks[k] = [(mask == 0)] * len(bbox_coords[k])


   #name = 'image_2023-07-06_14-01-52-172651_1'
   #maskname = 'mask_2023-07-06_14-01-52-172651_1'

    image_path = os.path.join(OutPreparedImages, f'image{k[4:]}.png')    
    image = cv2.imread(image_path)

    #image = cv2.imread(f'{OutPreparedImages}{f.stem}.png')

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    #fig, ax = plt.subplots()
    #show_boxes(bbox_coords[maskname], ax)
    #show_boxes(bbox_coords[maskname], plt.gca())
    show_boxes(bbox_coords[f.stem], plt.gca())


#    show_masks(ground_truth_masks[maskname], plt.gca())
    show_masks(ground_truth_masks[f.stem], plt.gca())    
    plt.axis('off')
    plt.show()

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
from collections import defaultdict

import torch

from segment_anything.utils.transforms import ResizeLongestSide

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
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
#keys = list(bbox_coords.keys())
keys = list(set(bbox_coords.keys()))   # Get unique list of keys 
############################################################################################

############################################################################################

num_epochs = 2
losses = []

for epoch in range(num_epochs):
  epoch_losses = []
  # Just train on the first 2 examples
  for k in keys[:3]:
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']
    
    # No grad here as we don't want to optimize the encoders
    with torch.no_grad():
      image_embedding = sam_model.image_encoder(input_image)
      
      prompt_boxes = bbox_coords[k]  # Multiple bounding boxes
      prompt_boxes1 = np.array(prompt_boxes)  # Convert to NumPy array
      boxes = transform.apply_boxes(prompt_boxes1, original_image_size)
      
######################################################################################

######################################################################################
# Convert the image tensor to a numpy array
      image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

      plt.figure(figsize=(10,10))
      # Display the image using matplotlib
      plt.imshow(image_np)
      plt.axis('off')  # Optional: Turn off axis ticks and labels
      # Draw bounding boxes on the image
# Draw bounding boxes on the image

      show_boxes(boxes, plt.gca())  
    #   for box in boxes:
    #       x, y, w, h = box
    #       rect = Rectangle((x, y), w, h, edgecolor='r', linewidth=2, facecolor='none')
    #       plt.gca().add_patch(rect)

      
      plt.show()


      #   # Resize the input image tensor to match the desired size
      # resized_input_image = F.interpolate(input_image.unsqueeze(0), size=input_size, mode='bilinear', align_corners=False)
      # resized_input_image = resized_input_image.squeeze(0)

      # # Convert the tensor back to numpy array for visualization
      # resized_image = resized_input_image.cpu().numpy().transpose(1, 2, 0)

      # # Draw bounding boxes on the resized image
      # for box in boxes:
      #     cv2.rectangle(resized_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

      # # Display the resized image with the bounding boxes
      # plt.imshow(resized_image)
      # plt.show()
      
######################################################################################

######################################################################################      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
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
      #  gt_mask_resized = torch.from_numpy(np.resize(gt_mask, (1, 1, gt_mask.shape[0], gt_mask.shape[1]))).to(device)
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


