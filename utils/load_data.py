import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

def horizontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
###############################################################################
class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)
###############################################################################
class GetDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        if 'train' in list_path:
            self.label_files = [
                    path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
                    for path in self.img_files
                    ]
        else:
            self.label_files = [
                    path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
                    for path in self.img_files
                    ]   
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        # IMAGE
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # extract image as PyTorch tensor
        imgs = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # handle images with less than three channels
        if len(imgs.shape) != 3:
            imgs = imgs.unsqueeze(0)
            imgs = imgs.expand((3, imgs.shape[1:]))
        _, h, w = imgs.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # pad to square resolution
        imgs, pad = pad_to_square(imgs, 0)
        _, padded_h, padded_w = imgs.shape
        # LABEL
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None
        if os.path.exists(label_path):
            # print('we have labels!')
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # return (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                imgs, targets = horizontal_flip(imgs, targets)
        return img_path, imgs, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # select new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets
    
    def __len__(self):
        return len(self.img_files)
###############################################################################