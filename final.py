from pyexpat import model
import XrayData
import CephaloXrayData
import torchvision.transforms as transforms
import numpy as np
from random import randint
import torch
import model as m
import torch.optim as optim
import torch.nn as nn
from time import time
from math import pi
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from pyramid import pyramid, stack, pyramid_transform
import sys
import cephaloConstants

"""
List image sizes with: identify -format "%i: %wx%h\n" *.jpg
These images have 2256x2304 instead of 2260x2304 size
rm 1234.jpg 1240.jpg 134.jpg 159.jpg 188.jpg 254.jpg 435.jpg 608.jpg 609.jpg 759.jpg 769.jpg 779.jpg 938.jpg 1107.jpg
"""

IMG_SIZE_ORIGINAL = {'width': 1935, 'height': 2400}
IMG_SIZE_ROUNDED_TO_64 = {'width': 1920, 'height': 2432}
IMG_TRANSFORM_PADDING = {'width': IMG_SIZE_ROUNDED_TO_64['width'] - IMG_SIZE_ORIGINAL['width'],
                        'height': IMG_SIZE_ROUNDED_TO_64['height']- IMG_SIZE_ORIGINAL['height']}

def rescale_point_to_original_size(point):
    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

def show_landmarks(image, landmarks, ground_truth=None):
    """Show image with landmarks"""
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r', label="Prediction")
    if ground_truth is not None:
        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=10, marker='.', c='b', label="Ground Truth")
    plt.figlegend('', ('Red', 'Blue'), 'center left')
    plt.pause(0.001)  # pause a bit so that plots are updated

FOLDS = 4
LANDMARKS = 19
LEVELS = 6

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("Please enter image file path as argument to be analyzed:")
        print(f"\t{sys.argv[0]} <file_path>")
        exit(0)
    
    image_path = sys.argv[1]

    models = []

    for l in range(LANDMARKS):
        l_model = []
        for f in range(FOLDS):
            l_model.append(m.load_model(LEVELS, f"big_hybrid_{l}_{f}"))
        
        models.append(l_model)

    
    
