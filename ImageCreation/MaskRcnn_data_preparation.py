## from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from detection import utils
from torchvision import transforms as T

myImgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
myMasks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))


mask = Image.open(myMasks)
# convert the PIL Image into a numpy array
mask = np.array(mask)
# instances are encoded as different colors
obj_ids = np.unique(mask)
# first id is the background, so remove it
obj_ids = obj_ids[1:]

# split the color-encoded mask into a set
# of binary masks
masks = mask == obj_ids[:, None, None]






# class PedestrianDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         current_directory = os.getcwd()
#         print(current_directory)
        
#        # print(os.path.join(root, "PNGImages"))
#       #  print(os.listdir(os.path.join(root, "PNGImages")))
        
#         print( os.path.join(current_directory, 'ImageCreation', 'tmp', 'PNGImages', ''))
    
        
#         # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#         # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
#         self.imgs = list(sorted(os.path.join(current_directory, 'ImageCreation', 'tmp', 'PNGImages', '')))
#         self.masks = list(sorted(os.path.join(current_directory, 'ImageCreation', 'tmp', 'PNGMasks', '')))

#     def __getitem__(self, idx):
#         # load images ad masks
#         img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
#         img = Image.open(img_path).convert("RGB")
#         # note that we haven't converted the mask to RGB,
#         # because each color corresponds to a different instance
#         # with 0 being background
#         mask = Image.open(mask_path)

#         mask = np.array(mask)
#         # instances are encoded as different colors
#         obj_ids = np.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]

#         # split the color-encoded mask into a set
#         # of binary masks
#         masks = mask == obj_ids[:, None, None]

#         # get bounding box coordinates for each mask
#         num_objs = len(obj_ids)
#         boxes = []
#         for i in range(num_objs):
#             pos = np.where(masks[i])
#             xmin = np.min(pos[1])
#             xmax = np.max(pos[1])
#             ymin = np.min(pos[0])
#             ymax = np.max(pos[0])
#             boxes.append([xmin, ymin, xmax, ymax])

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
#         masks = torch.as_tensor(masks, dtype=torch.uint8)

#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)
    
    
    
    
    
# def get_transform(train):
#     transforms = []
# # converts the image, a PIL image, into a PyTorch Tensor
# #    transforms.append(T.ToTensor())
#     transforms.append(T.PILToTensor())

#     if train:
#         # during training, randomly flip the training images
#         # and ground-truth for data augmentation
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

# # use our dataset and defined transformations
# #dataset = PedestrianDataset('PennFudanPed', get_transform(train=True))
# #dataset_test = PedestrianDataset('PennFudanPed', get_transform(train=False))

# dataset = PedestrianDataset('tmp', get_transform(train=True))
# #dataset_test = PedestrianDataset('PennFudanPed', get_transform(train=False))


# # split the dataset in train and test set
# torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# # define training and validation data loaders
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=2, shuffle=True, num_workers=2,
#     collate_fn=utils.collate_fn)

# # data_loader_test = torch.utils.data.DataLoader(
# #     dataset_test, batch_size=1, shuffle=False, num_workers=2,
# #     collate_fn=utils.collate_fn)
# print(data_loader)