import os
import time
import datetime
import torch
import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
from torch.autograd import Variable
from terminaltables import AsciiTable
# libraries
from utils.parse_config import parse_data_config
from utils.load_data import GetDataset, ImageFolder
from utils.utils import load_classes, evaluate, non_max_suppression, rescale_boxes, weights_init_normal
from models.yolo_attack import Darknet
import pickle
import torchvision
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else "cpu")
print('Device:', device)
epsilon = 0.2
ROOT_DIR = os.getcwd()
MODEL_DEF = os.path.join(ROOT_DIR, "config/yolov3-custom.cfg")
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "config/custom.data")
PRETRAINED_WEIGHTS = os.path.join(ROOT_DIR, "checkpoints/weight.pth")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
PRED_DIR = os.path.join(ROOT_DIR, "predictions")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)

class FastGradientSignMethod():
    def __init__(self, model, epsilon, min_val, max_val):
        self.model = model
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        
    def l2_norm(self, original_grads, eps=1e-24):
        reduction_indices = list(range(1, len(original_grads.shape)))
        return torch.sqrt(torch.sum(torch.mul(original_grads,original_grads),dim=reduction_indices,keepdim=True)) + eps

    def perturb(self, original_images, labels):
        # original_images: values are within self.min_val and self.max_val
        x = original_images.clone()
        x.requires_grad = True 

        self.model.eval()

        with torch.enable_grad():# this is used because somewhere in the code, calculation of the grad is disabled
            outputs = self.model(x)# make sure its in eval_mode
            loss, outputs = self.model(x, labels)
            grads = torch.autograd.grad(loss, x, only_inputs=True)[0]
            scaled_x_grad = grads / self.l2_norm(grads)
            x.data += self.epsilon * scaled_x_grad
            x.clamp(self.min_val, self.max_val)

        return x, (self.epsilon * scaled_x_grad)


def test(net, valid_path, class_names, batch_size=1, nms_thres=0.5, conf_thres=0.1, iou_thres=0.5, img_size=416):
    # iou thresshold for non-maximum suppression
    # object confidence threshold
    # iou threshold required to qualify as detected
    net.eval()   
    start_time_test = time.time()
    print("Compute mAP...")
    precision, recall, AP, f1, ap_class, outputlist = evaluate(
            net,
            path=valid_path,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            img_size=img_size,
            batch_size=batch_size)
    print("Average Precisions:", ap_class)
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
    print(f"mAP: {AP.mean()}")
    print('The testing took: ', time.time() - start_time_test,'seconds')

def generate_attacks(net, train_path, num_epochs, batch_size, attack, save=False):
    dataset = GetDataset(train_path, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=dataset.collate_fn)
    diff_list=[]
    adv_data_list=[]
    img_list=[]

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        adv_data, diff = attack.perturb(original_images=imgs, labels = targets)
        # in every mini batch stack n(batch size) information in the total list
        for i in range(len(imgs)):
            diff_list.append(diff[i].detach().cpu().numpy())
            adv_data_list.append(adv_data[i].detach().cpu().numpy())
            img_list.append(imgs[i].detach().cpu().numpy())
    if save:
        print('saving images')
        adv_examples = (adv_data_list, diff_list, img_list)
        with open('adv_examples.pickle', 'wb') as f:
            pickle.dump(adv_examples, f)
        return adv_data_list, diff_list, img_list

    return adv_data_list, diff_list, img_list

if __name__ == "__main__":

    data_config = parse_data_config(DATA_CONFIG_PATH)
    valid_path_adv = os.path.join(ROOT_DIR, "sample/sample_path.txt")#for big dataset
    class_names = load_classes(data_config["names"])
    print('the classes in the label',class_names)
    net = Darknet(MODEL_DEF).to(device)
    if PRETRAINED_WEIGHTS:
        if PRETRAINED_WEIGHTS.endswith(".pth"):
        	print('using pretrained weights ending with pth')
        	net.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))
        else:
        	print('using pretrained weights, but not ending with pth')
        	net.load_darknet_weights(PRETRAINED_WEIGHTS)
    else:
    	net.apply(weights_init_normal)
        
    attack = FastGradientSignMethod(model = net, epsilon = 0.2, min_val=0, max_val=1)
    adv_data, diff, imgs = generate_attacks(net, valid_path_adv, num_epochs=1, batch_size=1, 
                                      attack=attack, save = False)