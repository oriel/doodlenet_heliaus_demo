from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import torchvision
import os
import random
import glob

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from utils.module_list import *

from dataset.kaist import Kaist

import glob
import cv2

### class indexes for rtfnet/mfnet dataset and its translation to cityscapes:
# 0: unlabelled
# 1: car
# 2: person
# 3: bike
# 4: curve -> can be either sidewalk or curb new class
# 5: car stop {pole?}
# 6: guardrail (cerca/grade)
# 7: color cone
# 8: bump
    
### class indexes for cityscapes
# -1: unlabelled
# 0: road
# 1: sidewalk
# 2: building
# 3: wall
# 4: fence
# 5: pole
# 6: traffic light
# 7: traffic sign
# 8: vegetation
# 9: terrain
# 10: sky
# 11: person
# 12: rider
# 13: car
# 14: truck
# 15: bus
# 16: train
# 17: motorcycle
# 18: bicycle

### rtfnet + cityscapes

# 19: curve
# 20: guardrail
# 21: color cone
# 22: bump

def kaist_classdict(idx):
    classdict = {0: 'unlabelled',
                 1: 'person'}
    return classdict[idx]

# original class dictionary provided by denso

def create_heliaus_label_colormap_denso(merge_classes=True):
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 142] # car
  colormap[1] = [220, 20, 60] # pedestrian
  colormap[2] = [220, 20, 60] # people 
  colormap[3] = [0, 142, 142] # animal
  colormap[4] = [190, 153, 153] # special vehicle
  colormap[5] = [220, 20, 60] # cyclist
  colormap[6] = [0, 0, 142] # motorcyclist
  colormap[7] = [0, 0, 0] # unlabelled
  return colormap

def create_heliaus_label_colormap():
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 0] # unlabelled
  colormap[1] = [0, 0, 142] # car
  colormap[2] = [220, 20, 60] # pedestrian
  colormap[3] = [0, 142, 142] # animal
  colormap[4] = [220, 20, 60] # cyclist
  colormap[5] = [0, 0, 142] # motorcyclist
  colormap[6] = [128, 64, 128] # road
  return colormap


def heliaus_classdict_denso(idx):
    classdict = {0: 'car',
                 1: 'pedestrian',
                 2: 'people',
                 3: 'animal',
                 4: 'special_vehicules',
                 5: 'cyclist',
                 6: 'motorcyclist'}

    return classdict[idx]

# customized class dictionary, with merged classes, unlabeled and road class
def heliaus_classdict(idx):
    classdict = {0: 'unlabeled',
                 1: 'car',
                 2: 'pedestrian',
                 3: 'animal',
                 4: 'cyclist',
                 5: 'motorcyclist',
                 6: 'road'}

    return classdict[idx]


def rtfnet_cityscapes_classdict(idx):
    classdict = {0: 'road',
                 1: 'sidewalk',
                 2: 'building',
                 3: 'wall',
                 4: 'fence',
                 5: 'pole',
                 6: 'traffic light',
                 7: 'traffic sign',
                 8: 'vegetation',
                 9: 'terrain',
                 10: 'sky',
                 11: 'person',
                 12: 'rider',
                 13: 'car',
                 14: 'truck',
                 15: 'bus',
                 16: 'train',
                 17: 'motorcycle',
                 18: 'bicycle',
                 19: 'curve',
                 20: 'guardrail',
                 21: 'cone',
                 22: 'bump'}
    return classdict[idx]


def rtfnet_to_cityscapes_classmap(labels):
    classmap = {0:-1, 1: 13, 2: 11, 3: 18, 4:19, 6:20, 7:21, 8:22}
    for key, item in classmap.items():
        labels[labels == key] = item
    return labels

# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True, thermal_image=None, invalid_label=0, dataset=None, testing=False):
    # Random rescale image
    raw_w, raw_h = image.size
    if testing:
        scale_ratio = 1
        resized_size = (raw_h, raw_w)
        i, j = 0, 0
        h, w = crop_size
    else:
        scale_ratio = random.uniform(scale_size[0], scale_size[1])
        resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
        image = transforms_f.resize(image, resized_size, Image.BILINEAR)
        if label is not None:
            label = transforms_f.resize(label, resized_size, Image.NEAREST)

    if thermal_image is not None:
        thermal_image = transforms_f.resize(thermal_image, resized_size, Image.BILINEAR)
    
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        if label is not None:
            label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=invalid_label, padding_mode='constant')
        #label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=0  , padding_mode='constant')

        if thermal_image is not None:
            thermal_image = transforms_f.pad(thermal_image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=invalid_label, padding_mode='constant')

    # Cropping
    if not testing:
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    if label is not None:
        label = transforms_f.crop(label, i, j, h, w)
    if thermal_image is not None:
        thermal_image = transforms_f.crop(thermal_image, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
            if thermal_image is not None:
                thermal_image = thermal_image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            if label is not None:
                label = transforms_f.hflip(label)
            if thermal_image is not None:
                thermal_image = transforms_f.hflip(thermal_image)

            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    if label is not None:
        label = (transforms_f.to_tensor(label) * 255).long()
    #label[label == invalid_label] = 5 # 7 # invalid pixels are re-mapped to index -1


    #print(torch.unique(label))
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    #print("before imagenet normalization:")
    #print(image.min())
    #print(image.max())

    if thermal_image is not None:
        thermal_image = transforms_f.to_tensor(thermal_image)
        #thermal_image -= thermal_image.min()
        #thermal_image /= thermal_image.max()

    ### Imagenet normalization
    #image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #thermal_image = transforms_f.normalize(thermal_image, mean=[0.485], std=[0.229])

    if thermal_image is not None:
        image = torch.cat((image, thermal_image[0,:,:].unsqueeze(0)), 0)

    
    #print(torch.unique(label))
    #print(len(torch.unique(label)))
    if logits is not None:
        return image, label, logits
    else:
        return image, label


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation, invalid_label=255, dataset=None):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):

        if data.shape[1] == 4: # if this is RGBT, split it
            data_rgb = data[:,:3,:,:]
            data_thermal = data[:,3,:,:]

            data_thermal_pil = transforms_f.to_pil_image(data_thermal.cpu())

            data_pil, label_pil, logits_pil = tensor_to_pil(data_rgb[k], label[k], logits[k])
            aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                        augmentation=apply_augmentation, thermal_image=data_thermal_pil, invalid_label=invalid_label, dataset=dataset)

        else:
            data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
            aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                        augmentation=apply_augmentation, invalid_label=invalid_label, dataset=dataset)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans


# --------------------------------------------------------------------------------
# Define segmentation label re-mapping
# --------------------------------------------------------------------------------
def cityscapes_class_map(mask):
    # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = 255
    mask_map[np.isin(mask, [7])] = 0
    mask_map[np.isin(mask, [8])] = 1
    mask_map[np.isin(mask, [11])] = 2
    mask_map[np.isin(mask, [12])] = 3
    mask_map[np.isin(mask, [13])] = 4
    mask_map[np.isin(mask, [17])] = 5
    mask_map[np.isin(mask, [19])] = 6
    mask_map[np.isin(mask, [20])] = 7
    mask_map[np.isin(mask, [21])] = 8
    mask_map[np.isin(mask, [22])] = 9
    mask_map[np.isin(mask, [23])] = 10
    mask_map[np.isin(mask, [24])] = 11
    mask_map[np.isin(mask, [25])] = 12
    mask_map[np.isin(mask, [26])] = 13
    mask_map[np.isin(mask, [27])] = 14
    mask_map[np.isin(mask, [28])] = 15
    mask_map[np.isin(mask, [31])] = 16
    mask_map[np.isin(mask, [32])] = 17
    mask_map[np.isin(mask, [33])] = 18
    return mask_map


def sun_class_map(mask):
    # -1 is equivalent to 255 in uint8
    return mask - 1


# --------------------------------------------------------------------------------
# Define indices for labelled, unlabelled training images, and test images
# --------------------------------------------------------------------------------
def get_pascal_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_name = root + '/train_aug.txt'
    else:
        file_name = root + '/val.txt'
    with open(file_name) as f:
        idx_list = f.read().splitlines()

    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(21)
        label_fill = np.arange(21)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []
            mask = np.array(Image.open(root + '/SegmentationClassAug/{}.png'.format(idx)))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            unique_num = len(mask_unique)   # number of unique classes

            # sample image if it includes the lowest appeared class and with more than 3 distinctive classes
            if len(labeled_idx) == 0 and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            else:
                save_idx.append(idx)

            # record any segmentation index with lowest appearance
            label_fill = np.where(label_counter == label_counter.min())[0]

        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
    else:
        return idx_list


def get_kaist_idx(root, train=True, label_num=5):

    rgb_images = sorted(glob.glob(f"{root}/images/*/*/visible/*"))[:7560]
    t_images = sorted(glob.glob(f"{root}/images/*/*/lwir/*"))[:7560]
    labels = sorted(glob.glob(f"{root}/annotations/*/*/*"))[:7560]
    if train:

        rgb_images = rgb_images[:int(.8 * len(rgb_images))]
        t_images = t_images[:int(.8 * len(t_images))]
        labels = labels[:int(.8 * len(t_images))]

    else:
        rgb_images = rgb_images[int(.8 * len(rgb_images)):]
        t_images = t_images[int(.8 * len(t_images)):]
        labels = labels[int(.8 * len(t_images)):]

    num_samples = len(rgb_images)

    #TODO: take into account labeled / unlabeled split - label_num

    return list(range(num_samples)), list(range(num_samples))


def get_rtfnet_idx(root, train=True, label_num=5, min_classes = 2, randomize=True):

    if train:
        idx_filename = f'{root}/train.txt'
    else:
        #idx_filename = f'{root}/test.txt'
        idx_filename = f'{root}/val.txt'

    with open(idx_filename) as fp:
        lines = fp.readlines()
    idx_list = [l.replace('\n','') for l in lines if 'flip' not in l]
    
    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        if randomize:
            random.shuffle(idx_list_)
        label_counter = np.zeros(24)
        label_fill = np.arange(24)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []

            if randomize:
                if random.randint(0,1) == 1:
                    labeled_idx.append(idx)
                else:
                    save_idx.append(idx)
            else:
                # sample image if it includes the lowest appeared class and with more than min_classes distinctive classes
                mask = cityscapes_class_map(np.array(Image.open(root + '/labels/train/{}.png'.format(idx))))
                mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
                unique_num = len(mask_unique)  # number of unique classes
                if len(labeled_idx) == 0 and unique_num >= min_classes:
                    labeled_idx.append(idx)
                    label_counter[mask_unique] += 1
                elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= min_classes:
                    labeled_idx.append(idx)
                    label_counter[mask_unique] += 1
                else:
                    save_idx.append(idx)
                # record any segmentation index with lowest occurrence
                label_fill = np.where(label_counter == label_counter.min())[0]
            
        non_labeled_idx = [idx for idx in idx_list if idx not in labeled_idx]
        print(f"Number of training labeled images: {len(labeled_idx)}")
        print(f"Number of training non-labeled images: {len(non_labeled_idx)}")

        return labeled_idx, non_labeled_idx
    else:
        return idx_list

    

    #self.idx_list = list(range(len(self.rgb_images)))

    
    #rgb_images = sorted(glob.glob(f"{root}/separated_images/*rgb.png"))
    #t_images = sorted(glob.glob(f"{root}/separated_images/*th.png"))
    #labels = sorted(glob.glob(f"{root}/labels/*D.png") + glob.glob(f"{root}/labels/*N.png"))
    #labels = sorted(glob.glob(f"{root}/labels/*.png"))

    #self.idx_list = list(range(len(self.rgb_images)))
    # while len(labeled_idx) < label_num:
    #     if len(idx_list_) > 0:
    #         idx = idx_list_.pop()
    #     else:
    #         idx_list_ = save_idx.copy()
    #         idx = idx_list_.pop()
    #         save_idx = []

    #um_samples = len(rgb_images)

    #TODO: take into account labeled / unlabeled split - label_num

    #eturn list(range(num_samples)), list(range(num_samples))


def get_cityscapes_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + '/images/train/*.png')
    else:
        file_list = glob.glob(root + '/images/val/*.png')
    idx_list = [int(file[file.rfind('/') + 1: file.rfind('.')]) for file in file_list]

    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(19)
        label_fill = np.arange(19)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []

            mask = cityscapes_class_map(np.array(Image.open(root + '/labels/train/{}.png'.format(idx))))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            unique_num = len(mask_unique)  # number of unique classes

            # sample image if it includes the lowest appeared class and with more than 12 distinctive classes
            if len(labeled_idx) == 0 and unique_num >= 12:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 12:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            else:
                save_idx.append(idx)

            # record any segmentation index with lowest occurrence
            label_fill = np.where(label_counter == label_counter.min())[0]
            
        non_labeled_idx = [idx for idx in idx_list if idx not in labeled_idx]
        return labeled_idx, non_labeled_idx
    else:
        return idx_list


def get_sun_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + '/SUNRGBD-train_images/*.jpg')
        idx_list = [int(file[file.rfind('-') + 1: file.rfind('.')]) for file in file_list]
    else:
        file_list = glob.glob(root + '/SUNRGBD-test_images/*.jpg')
        idx_list = [int(file[file.rfind('-') + 1: file.rfind('.')]) for file in file_list]

    if train:
        labeled_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)

        # create a label dictionary class_list [[Image_ID for class 0], [Image_ID for class 1], ...]
        class_list = [[] for _ in range(37)]
        for i in range(len(idx_list_)):
            idx = idx_list_[i]
            mask = sun_class_map(np.array(Image.open(root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(idx))))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            for k in mask_unique:
                class_list[k].append(idx)

        label_counter = np.zeros(37)
        label_fill = np.arange(37)
        ignore_val = []
        ignore_mask = np.ones(37, dtype=bool)  # store any semantic id has sampled all possible images
        while len(labeled_idx) < label_num:
            if len(class_list[label_fill[0]]) > 0:
                idx = class_list[label_fill[0]].pop()
            else:
                ignore_val.append(label_fill[0])
                ignore_mask[ignore_val] = False

            # sample image by the current lowest appeared class
            if idx not in labeled_idx:
                labeled_idx.append(idx)
                mask = sun_class_map(np.array(Image.open(root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(idx))))
                mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
                label_counter[mask_unique] += 1

                # record any segmentation index with lowest occurrence
                label_fill = np.where(label_counter == label_counter[ignore_mask].min())[0]

            ignore_ind = [np.where(label_fill == i) for i in ignore_val if i in label_fill]
            if len(ignore_ind) > 0:  # ignore index when reaching all available images
                label_fill = np.delete(label_fill, ignore_ind)

        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
    else:
        return idx_list


# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed

        

        if 'rtfnet' in dataset:

            self.rgb_images = [f"separated_images/{im}_rgb.png" for im in idx_list]
            self.t_images = [f"separated_images/{im}_th.png" for im in idx_list]
            self.labels = [f"labels/{im}.png" for im in idx_list]

            #rgb_images = sorted(glob.glob(f"{root}/separated_images/*rgb.png"))
            #t_images = sorted(glob.glob(f"{root}/separated_images/*th.png"))
            #labels = sorted(glob.glob(f"{root}/labels/*D.png") + glob.glob(f"{root}/labels/*N.png"))
                        
        if  'kaist' in dataset:
            rgb_images = sorted(glob.glob("{}/images/*/*/visible/*".format(root)))
            t_images = sorted(glob.glob("{}/images/*/*/lwir/*".format(root)))
            labels = sorted(glob.glob("{}/annotations/*/*/*".format(root)))
            
            if train:
                self.rgb_images = rgb_images[:int(.8 * len(rgb_images))]
                self.t_images = t_images[:int(.8 * len(t_images))]
                self.labels = labels[:int(.8 * len(t_images))]

            else:
                self.rgb_images = rgb_images[int(.8 * len(rgb_images)):]
                self.t_images = t_images[int(.8 * len(t_images)):]
                self.labels = labels[int(.8 * len(t_images)):]



    def __getitem__(self, index):
        if self.dataset == 'pascal':
            image_root = Image.open(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[index]))
            if self.apply_partial is None:
                label_root = Image.open(self.root + '/SegmentationClassAug/{}.png'.format(self.idx_list[index]))
            else:
                label_root = Image.open(self.root + '/SegmentationClassAug_{}_{}/{}.png'.format(self.apply_partial,  self.partial_seed, self.idx_list[index],))

            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

        if self.dataset == 'cityscapes':
            if self.train:
                image_root =  Image.open(self.root + '/images/train/{}.png'.format(self.idx_list[index]))
                if self.apply_partial is None:
                    label_root = Image.open(self.root + '/labels/train/{}.png'.format(self.idx_list[index]))
                else:
                    label_root = Image.open(self.root + '/labels/train_{}_{}/{}.png'.format(self.apply_partial,  self.partial_seed, self.idx_list[index]))
                label_root = Image.fromarray(cityscapes_class_map(np.array(label_root)))
            else:
                image_root = Image.open(self.root + '/images/val/{}.png'.format(self.idx_list[index]))
                label_root = Image.open(self.root + '/labels/val/{}.png'.format(self.idx_list[index]))
                label_root = Image.fromarray(cityscapes_class_map(np.array(label_root)))

            
            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)


        if 'rtfnet' in self.dataset:
            #import random
            #index = random.randint(0,100)
            #print("index:") 
            #print(index)
            image_rgb_root = Image.open(os.path.join(self.root, self.rgb_images[index]))
            w, h = image_rgb_root.size
            image_annotations = Image.open(os.path.join(self.root, self.labels[index]))
            
            label_root = Image.fromarray(np.uint8(image_annotations))

            if 'rgbt' in self.dataset:
                image_t_root = Image.open(os.path.join(self.root, self.t_images[index]))
            else:
                image_t_root = None
            
            image, label = transform(image_rgb_root, label_root, None, self.crop_size, self.scale_size, self.augmentation, thermal_image=image_t_root, dataset=self.dataset)
            #print(image.shape)
            #print(np.unique(label))

            if "train" in self.rgb_images[index] or "val" in self.rgb_images[index]:
                label = torch.tensor(cityscapes_class_map(label))
            else:
                label = rtfnet_to_cityscapes_classmap(label)
                #print("found cityscape data, converted labels:")
                #print(np.unique(label))
                #exit()
            label[label == 255] = -1  # invalid pixels are re-mapped to index -1
            #print(type(image))
            return image, label.squeeze(0)                


        if 'kaist' in self.dataset:

            image_rgb_root = Image.open(self.rgb_images[index])
            w, h = image_rgb_root.size
            image_annotations = np.zeros((h, w)) #-1
            
            with open(self.labels[index]) as fp:
                annotations = fp.readlines()[1:]
                if annotations != []:
                    for annotation in annotations:
                        cat, x, y, w, h = annotation.split(" ")[:5]
                        #print(cat)
                        if cat == "person" or cat == "people":
                            #image_annotations[int(y):int(y)+int(h), int(x):int(x)+int(w)] = 12 # 12 is the idx for person class in cityscapes
                            image_annotations[int(y):int(y)+int(h), int(x):int(x)+int(w)] = 1

            label_root = Image.fromarray(np.uint8(image_annotations))

            if 'rgbt' in self.dataset:
                image_t_root = Image.open(self.t_images[index])
            else:
                image_t_root = None
            
            image, label = transform(image_rgb_root, label_root, None, self.crop_size, self.scale_size, self.augmentation, thermal_image=image_t_root)
            
            return image, label.squeeze(0)                

            #     tr_gen = Kaist(root=hparams.data_folder, split=split, paired=True, transforms=pipe, use_thermal=hparams.use_thermal)
            # if split == 'train':
            #     loader = DataLoader(tr_gen, **loader_params)
            # elif split == 'valid':
            #     split = 'val'
            #     loader_params['shuffle']=False
            #     loader = [DataLoader(tr_gen, **loader_params), DataLoader(tr_gen, **loader_params)]


        if self.dataset == 'sun':
            if self.train:
                image_root = Image.open(self.root + '/SUNRGBD-train_images/img-{:06d}.jpg'.format(self.idx_list[index]))
                label_root = Image.open(self.root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(self.idx_list[index]))
                label_root = Image.fromarray(sun_class_map(np.array(label_root)))
            else:
                image_root = Image.open(self.root + '/SUNRGBD-test_images/img-{:06d}.jpg'.format(self.idx_list[index]))
                label_root = Image.open(self.root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(self.idx_list[index]))
                label_root = Image.fromarray(sun_class_map(np.array(label_root)))
            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

    def __len__(self):
        return len(self.idx_list)


# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataLoader:
    def __init__(self, dataset, num_labels):
        self.dataset = dataset
        if dataset == 'pascal':
            self.data_path = 'dataset/pascal'
            self.im_size = [513, 513]
            self.crop_size = [321, 321]
            self.num_segments = 21
            self.scale_size = (0.5, 1.5)
            self.batch_size = 10
            self.train_l_idx, self.train_u_idx = get_pascal_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_pascal_idx(self.data_path, train=False)

        if dataset == 'cityscapes':
            #self.data_path = 'dataset/cityscapes'
            self.data_path = '/home/ofrigo/datasets/cityscapes'
            self.im_size = [512, 1024]
            self.crop_size = [512, 512]
            self.num_segments = 19
            self.scale_size = (1.0, 1.0)
            self.batch_size = 2
            self.train_l_idx, self.train_u_idx = get_cityscapes_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_cityscapes_idx(self.data_path, train=False)
            self.invalid_label = 255


        if  "kaist" in dataset:
            self.data_path = '/home/ofrigo/datasets/kaist/SoonminHwang-rgbt-ped-detection-50ac9b3/data/kaist-rgbt'
            #self.data_path = 'dataset/kaist-rgbt'
            #check_kaist_available = Kaist(root=self.data_path)
            self.im_size = [512, 640]
            self.crop_size = [512, 512]
            #self.num_segments = 19
            self.num_segments = 2            
            self.scale_size = (1.0, 1.0)
            self.batch_size = 5
            self.train_l_idx, self.train_u_idx = get_kaist_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_kaist_idx(self.data_path, train=False)            
            print("loading kaist")

        if 'heliaus' in dataset:
            #from dataset.heliaus import Heliaus
            #dataset = Heliaus(root="/mnt/heliaus/", category=['Daytime', 'Nighttime'], split='val', supervised=True, transform=None)

            #rgb_pil, lwi_pil, ty = dataset[-1]
            #image, label = transform(rgb_pil, ty, None, crop_size=(512,512), scale_size=(1.0, 1.0), augmentation=False, thermal_image=None)
            # print(np.unique(image))
            #print(np.unique(label))
            #exit()

            self.im_size = [480, 640]
            self.crop_size = [480, 640]
            #self.num_segments = 7
            self.num_segments = 8
            self.scale_size = (1.0, 1.0)
            self.batch_size = 2
            #self.train_l_idx, self.train_u_idx = get_rtfnet_idx(self.data_path, train=True, label_num=num_labels)
            #self.test_idx = get_rtfnet_idx(self.data_path, train=False)
            self.invalid_label = -1
            print("loading heliaus dataset")

            

        if 'rtfnet' in dataset:
            #self.data_path = 'dataset/rtfnet/dataset/'
            #self.data_path = 'dataset/cityscapes_rtfnet/'
            self.data_path = "/home/ofrigo/datasets/rfnet/dataset"
            self.im_size = [512, 640]
            self.crop_size = [512, 512]
            self.num_segments = 23
            #self.num_segments = 9
            self.scale_size = (1.0, 1.0)
            self.batch_size = 2
            self.train_l_idx, self.train_u_idx = get_rtfnet_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_rtfnet_idx(self.data_path, train=False)
            self.invalid_label = 0
            print("loading rtfnet dataset")

        if dataset == 'sun':
            self.data_path = 'dataset/sun'
            self.im_size = [385, 513]
            self.crop_size = [321, 321]
            self.num_segments = 37
            self.scale_size = (0.5, 1.5)
            self.batch_size = 5
            self.train_l_idx, self.train_u_idx = get_sun_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_sun_idx(self.data_path, train=False)

        if num_labels == 0 and 'heliaus' not in dataset:  # using all data
            self.train_l_idx = self.train_u_idx

    def build(self, supervised=False, partial=None, partial_seed=None):
        train_l_dataset = BuildDataset(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)
        train_u_dataset = BuildDataset(self.data_path, self.dataset, self.train_u_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=True, apply_partial=partial, partial_seed=partial_seed)
        test_dataset    = BuildDataset(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        if supervised:  # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        num_samples = self.batch_size * 200  # for total 40k iterations with 200 epochs

        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            num_workers=5,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
        )

        if not supervised:
            train_u_loader = torch.utils.data.DataLoader(
                train_u_dataset,
                batch_size=self.batch_size,
                sampler=sampler.RandomSampler(data_source=train_u_dataset,
                                              replacement=True,
                                              num_samples=num_samples),
                drop_last=True,
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
        )
        if supervised:
            return train_l_loader, test_loader
        else:
            return train_l_loader, train_u_loader, test_loader

        

# --------------------------------------------------------------------------------
# Create Color-mapping for visualisation
# --------------------------------------------------------------------------------
def create_cityscapes_label_colormap():
  """Creates a label colormap used in CityScapes segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[-1] = [0, 0, 0] # unlabelled
  colormap[0] = [128, 64, 128] # road
  colormap[1] = [244, 35, 232] # sidewalk
  colormap[2] = [70, 70, 70] # building
  colormap[3] = [102, 102, 156] # wall
  colormap[4] = [190, 153, 153] # fence
  colormap[5] = [153, 153, 153] # pole
  colormap[6] = [250, 170, 30] # traffic light
  colormap[7] = [220, 220, 0] # traffic sign
  colormap[8] = [107, 142, 35] # vegetation
  colormap[9] = [152, 251, 152] # terrain
  colormap[10] = [70, 130, 180] # sky
  colormap[11] = [220, 20, 60] # person
  colormap[12] = [255, 0, 0] # rider
  colormap[13] = [0, 0, 142] # car
  colormap[14] = [0, 0, 70] # truck
  colormap[15] = [0, 60, 100] # bus
  colormap[16] = [0, 80, 100] # train
  colormap[17] = [0, 0, 230] # motorcycle
  colormap[18] = [119, 11, 32] # bicycle
  return colormap


def create_rtfnet_label_colormap(colormap):
  """Creates a label colormap based in CityScapes segmentation benchmark combined with rtfnet.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap[19] = [0, 0, 192] 
  colormap[20] = [64, 64, 128] 
  colormap[21] = [192, 128, 128] 
  colormap[22] = [192, 64, 0] 
  return colormap


def create_pascal_label_colormap():
  """Creates a label colormap used in Pascal segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = 255 * np.ones((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 0]
  colormap[1] = [128, 0, 0]
  colormap[2] = [0, 128, 0]
  colormap[3] = [128, 128, 0]
  colormap[4] = [0, 0, 128]
  colormap[5] = [128, 0, 128]
  colormap[6] = [0, 128, 128]
  colormap[7] = [128, 128, 128]
  colormap[8] = [64, 0, 0]
  colormap[9] = [192, 0, 0]
  colormap[10] = [64, 128, 0]
  colormap[11] = [192, 128, 0]
  colormap[12] = [64, 0, 128]
  colormap[13] = [192, 0, 128]
  colormap[14] = [64, 128, 128]
  colormap[15] = [192, 128, 128]
  colormap[16] = [0, 64, 0]
  colormap[17] = [128, 64, 0]
  colormap[18] = [0, 192, 0]
  colormap[19] = [128, 192, 0]
  colormap[20] = [0, 64, 128]
  return colormap


def create_sun_label_colormap():
  """Creates a label colormap used in SUN RGB-D segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [148, 65, 137]
  colormap[1] = [255, 116, 69]
  colormap[2] = [86, 156, 137]
  colormap[3] = [202, 179, 158]
  colormap[4] = [155, 99, 235]
  colormap[5] = [161, 107, 108]
  colormap[6] = [133, 160, 103]
  colormap[7] = [76, 152, 126]
  colormap[8] = [84, 62, 35]
  colormap[9] = [44, 80, 130]
  colormap[10] = [31, 184, 157]
  colormap[11] = [101, 144, 77]
  colormap[12] = [23, 197, 62]
  colormap[13] = [141, 168, 145]
  colormap[14] = [142, 151, 136]
  colormap[15] = [115, 201, 77]
  colormap[16] = [100, 216, 255]
  colormap[17] = [57, 156, 36]
  colormap[18] = [88, 108, 129]
  colormap[19] = [105, 129, 112]
  colormap[20] = [42, 137, 126]
  colormap[21] = [155, 108, 249]
  colormap[22] = [166, 148, 143]
  colormap[23] = [81, 91, 87]
  colormap[24] = [100, 124, 51]
  colormap[25] = [73, 131, 121]
  colormap[26] = [157, 210, 220]
  colormap[27] = [134, 181, 60]
  colormap[28] = [221, 223, 147]
  colormap[29] = [123, 108, 131]
  colormap[30] = [161, 66, 179]
  colormap[31] = [163, 221, 160]
  colormap[32] = [31, 146, 98]
  colormap[33] = [99, 121, 30]
  colormap[34] = [49, 89, 240]
  colormap[35] = [116, 108, 9]
  colormap[36] = [161, 176, 169]
  return colormap


def create_nyuv2_label_colormap():
  """Creates a label colormap used in NYUv2 segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 255]
  colormap[1] = [233,  89,  48]
  colormap[2] = [0, 218,  0]
  colormap[3] = [149, 0, 240]
  colormap[4] = [222, 241,  24]
  colormap[5] = [255, 206, 206]
  colormap[6] = [0, 224., 229]
  colormap[7] = [10., 136., 204]
  colormap[8] = [117,  29,  41]
  colormap[9] = [240,  35, 235]
  colormap[10] = [0, 167, 156.]
  colormap[11] = [249, 139,   0]
  colormap[12] = [225, 229, 194]
  return colormap


def color_map(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)


def visualize(writer, pred, data, labels, title, colormap, idx, draw_confidence=True):
    
    nrow = 3
    pad_value = 0.2
    b, c, h, w = data.size()

    if c == 4:
        nrow += 1

    if draw_confidence:
        nrow += 1
        
    train_viz = torch.zeros(b, nrow, 3, h, w)

    pred_logits = F.interpolate(pred, size=data.shape[2:], mode='bilinear', align_corners=True)                        
    max_logits, label_reco = torch.max(torch.softmax(pred_logits, dim=1), dim=1)
    #label_reco[max_logits < 0.5] = -1

    for b in range(len(data)):

        #if args.dataset == 'cityscapes':
        #label_reco[b][train_l_label[b] == -1] = -1

        if c == 3:
            incr = 0
            train_viz[b, 0, :, :, :] = (data[b]) #- data[b].min())/(data[b].max() - data[b].min())
            mean_img = data[b, :3].cpu().numpy()
        elif c == 4:
            train_viz[b, 0, 0, :, :] = (data[b,0]) #- data[b,0].min())/(data[b,0].max() - data[b,0].min())
            train_viz[b, 0, 1, :, :] = (data[b,1]) #- data[b,1].min())/(data[b,1].max() - data[b,1].min())
            train_viz[b, 0, 2, :, :] = (data[b,2]) #- data[b,2].min())/(data[b,2].max() - data[b,2].min())
            train_viz[b, 1, :, :, :] = data[b,3]
            incr = 1
            rgb_img = data[b, :3].cpu().numpy()
            thermal_img = data[b, 3].cpu().numpy()
            mean_img = (0.6*rgb_img + 0.4*thermal_img)

        pred_labels_img = color_map(label_reco[b].cpu(), colormap)/255.
        gt_labels_img = color_map(labels[b].cpu(), colormap)/255.
        
        pred_labels_blended = np.zeros_like(mean_img) 
        gt_labels_blended = np.zeros_like(mean_img)

        for k in range(3):
            pred_labels_blended[k] = 0.4*pred_labels_img[:,:,k] + 0.6*mean_img[k]
            gt_labels_blended[k] = 0.4*gt_labels_img[:,:,k] + 0.6*mean_img[k]

        #train_viz[b, 1+incr, :, :, :] = torch.tensor(pred_labels_img).permute(2,0,1)
        #train_viz[b, 2+incr, : ,:, :] = torch.tensor(gt_labels_img).permute(2,0,1)

        train_viz[b, 1+incr, :, :, :] = torch.tensor(pred_labels_blended)
        train_viz[b, 2+incr, : ,:, :] = torch.tensor(gt_labels_blended)

        #import pdb
        #pdb.set_trace()
        
        if draw_confidence:
            # draw confidence map
            train_viz[b, 3+incr, :, :, :] = max_logits[b]

        
    train_viz_flat = train_viz.view(-1, 3, h, w)
    grid_img = torchvision.utils.make_grid(train_viz_flat, nrow=nrow, pad_value=pad_value)

    writer.add_image(
        title,
        grid_img,
        idx * len(data),
    )

def get_classmap(dataset):
    if "kaist" in dataset:
        colormap = create_cityscapes_label_colormap()
        classdict = kaist_classdict
    if "cityscapes" in dataset:
        colormap = create_cityscapes_label_colormap()
        classdict = rtfnet_cityscapes_classdict 
    if "heliaus" in dataset:
        colormap = create_heliaus_label_colormap()
        classdict = heliaus_classdict 
    if "rtfnet" in dataset:
        colormap = create_cityscapes_label_colormap()
        colormap = create_rtfnet_label_colormap(colormap)
        classdict = rtfnet_cityscapes_classdict
    return colormap, classdict
