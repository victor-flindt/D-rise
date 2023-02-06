import torch
import numpy as np 
import matplotlib.pyplot as plt
#from torchsummary import summary
#summary(model,(3,640,480))
import cv2
### yolov5 specific ultil functions
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
### To load yolov5s
from models.common import DetectMultiBackend
%matplotlib inline
import math
from tqdm import tqdm
import sys
import PIL
from PIL import Image
from pathlib import Path
import pandas as pd

#for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import time


def inference_batch(model_path,image,conf_tresh=0.5,max_det=1000,apply_bbox = True):
    """_summary_

    Args:
        model_path (__path__): path to model.pt weight file.
        image_path (__path__): Image tensor [batch,h,w,channels] expected.

    Returns:
        np.array: image with bbox
        tensor  : 
    """
    device = select_device('')
    model = DetectMultiBackend(f'{model_path}', device=device)
    model.eval()
    ### Prediction with model
    with torch.no_grad():
        img_np = image
        img = img_np.float().to(device).permute(0,3,1,2)


        ### Normalize the input, else everything is a detection
        img /= 255
        ### Prediction with model
        preds = model(img)
        ### Non max supression with conf and max detection arguments
        # print(preds)
        # print(type(preds))
        # print(preds[0].shape)
        predictions = non_max_suppression(preds, conf_thres = conf_tresh, max_det=max_det)
        names = model.names
        return_predictions = []
        return_images = []
        img_np = img_np.cpu().detach().numpy()
        for index,prediction in enumerate(predictions):
            if ((len(prediction)!=0)&(apply_bbox==True)):
                for pred in prediction:
                    ### instead of squeezing using [0] to avoid extra dim.
                    x1,y1,x2,y2,conf,id = np.asarray(pred.cpu())
                    start_point = (int(x1),int(y1))
                    end_point = (int(x2),int(y2))
                    color = (255, 0, 0)
                    ### Drawing prediction bbox
                    img_return = cv2.rectangle(img_np[index],start_point,end_point,color=color,thickness=2)
                    ### Drawing the label
                    name_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    origin = (int(x1),int(y1-10))
                    img_return = cv2.putText(img_return,f'{names[int(id)]}: {conf:0f}',origin,name_font,0.7,color,1)
                    return_predictions.append(pred)
                    return_images.append(img_return)
            else:
                return_predictions.append(prediction)
                return_images.append(img_np[index])
    return return_predictions,return_images

def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask
    
def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked
def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)

def generate_saliency_map(image,
                          target_class_index,
                          model_path,
                          target_box,
                          prob_thresh=0.5,
                          grid_size=(8, 8),
                          n_runs= 20,
                          n_masks=200,
                          seed=0
                          ):
    """_summary_

    Args:
        image (tensor): tensor image
        target_class_index (int): int representing the target class
        model_path (path/str): path to model
        target_box (x1,y1,x2,y2): bbox of the ground truth box
        prob_thresh (float, optional): _description_. Defaults to 0.5.
        grid_size (tuple, optional): _description_. Defaults to (8, 8).
        n_runs (int, optional): nr. of repetitions with n_mask, enables batching inference without overloading GPU. Defaults to 20.
        n_masks (int, optional): _description_. Defaults to 200.
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    np.random.seed(seed)
    image_h, image_w = image.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    for __ in tqdm(range(n_runs)):
        batch = []
        mask_list = []
        for _ in range(n_masks):
            mask = generate_mask(image_size=(image_w, image_h),
                                grid_size=grid_size,
                                prob_thresh=prob_thresh)
            mask_list.append(mask)
            masked = mask_image(image, mask)
            batch.append(torch.tensor(masked))
        ### Converting list of tensor images to batch, this is by far the fastes way
        batch = torch.stack(batch,dim=0)
        preds,_ = inference_batch(model_path,batch)
        ### itterating over all the images and predictions
        for index,(all_preds_mask_n,mask_n) in enumerate(zip(preds,mask_list)):
            for prediction in [all_preds_mask_n]:
                if (len(prediction)!=0)and(prediction[5] == target_class_index):
                    box = tuple(prediction[0:4].cpu().numpy().astype(int).tolist())
                    score = float(prediction[4].cpu())
                    temp_pred = (box,score)
                    score = max([iou(target_box, box) * score for box, score in [temp_pred]],
                    default=0)
                    res += mask_n * score
    return res
    
def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

runs = 10
n_mask = 500

program_starts = time.time()
label_names     = ['scratch','hair','paint_splatter','paint_spray']
model_path      = 'runs/train/exp07_10_2022_13_30_16/weights/best.pt'
image_path      = '/home/ubuntu/VTGF_cand_project/datasets/class_examples/scratch/PdsHv5_M1St22-7-1_Fail_0537.bmp'
image           = cv2.imread(image_path)
gt_path         = f'/home/ubuntu/VTGF_cand_project/datasets/class_examples/scratch/PdsHv5_M1St22-7-1_Fail_0537.txt'
h_image,w_image = image.shape[0:2]
ground_truth    = pd.read_csv(gt_path,header=None,delimiter=" ").values.tolist()

ground_truth_class = int(ground_truth[0][0])
center_X = ground_truth[0][1]
center_Y = ground_truth[0][2]
w_bbox   = ground_truth[0][3]
h_bbox   = ground_truth[0][4]

x1 = (center_X-w_bbox/2)*w_image
x2 = (center_X+w_bbox/2)*w_image
y1 = (center_Y-h_bbox/2)*h_image
y2 = (center_Y+h_bbox/2)*h_image

target_box = (int(x1),int(y1),int(x2),int(y2))

target_class_index  = ground_truth_class
saliency_map = generate_saliency_map(image,
                                    target_class_index=target_class_index,
                                    model_path=model_path,
                                    target_box=target_box,
                                    prob_thresh=0.5,
                                    grid_size=(16,16),
                                    n_runs = runs,
                                    n_masks=n_mask)
                                    
now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))

### Displaying image
    #cv2.rectangle(image, tuple(target_box[:2]), tuple(target_box[2:]),(0, 255, 0), 5)
plt.imshow(image[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
