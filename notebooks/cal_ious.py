import pandas as pd
import os
import cv2
import math
import json
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import namedtuple
from matplotlib import pyplot as plt
import itertools


BODY_CLASSES = [
    "background",
    "foot",
    "hand",
    "arm",
    "leg",
    "torso",
    "head"
]

ANN_ROOT_DIR = "/data/sara/semantic-segmentation-pytorch/"

random.seed(0)
NUM_CLASSES = 7
BODY_COLORMAP = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(NUM_CLASSES)]


def transparent_overlays(image, annotation, alpha=0.5):
    img1 = image.copy()
    img2 = annotation.copy()

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg, img2_fg)
    dst = cv2.addWeighted(image.copy(), 1-alpha, img2_fg, alpha, 0)
    img1[0:rows, 0:cols ] = dst
    return dst

def color_im(img, colors):
    for i in range(len(colors)):
        img[:,:,0][np.where(img[:,:,0] == i)] = colors[i][0]
        img[:,:,1][np.where(img[:,:,1] == i)] = colors[i][1]
        img[:,:,2][np.where(img[:,:,2] == i)] = colors[i][2]
    return img

def vis_pair(img1, img2, label1, label2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    label1 = cv2.imread(label1)

    label2 = cv2.imread(label2)

    tmp_label2 = cv2.resize(label2, (int(label1.shape[1]), int(label1.shape[0])))

    label1[np.where(label1 == 0)] = 255 # to exclude the bg from intersection

    intersection = np.where(label1[:,:,0] == tmp_label2[:,:,0])[0].shape[0]
    union = np.where(label1[:,:,0] != 255 )[0].shape[0] + np.where(tmp_label2[:,:,0] != 0 )[0].shape[0] - intersection
    '''
    plt.figure(figsize=(10, 7))
    plt.imshow(img1)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.imshow(img2)
    plt.show()


    plt.figure(figsize=(10, 7))
    plt.imshow(transparent_overlays(color_im(label1, colors), color_im(tmp_label2, colors), 0.45))
    plt.show()
    '''

    return intersection/union
df = pd.read_csv("../data/bodyparts_csv/train.csv")
colors = BODY_COLORMAP

output = ""

for i, c in enumerate(BODY_CLASSES):
    if c != 'background':
        #print(c)
        sub_data = df[df[c]==1]
        #print(f"numbers of image: {len(sub_data)} \n\n")
        count = 0
        for pair in itertools.combinations(sub_data['fpath_img'],2):
            row1 = df[df['fpath_img']==pair[0]]
            row2 = df[df['fpath_img']==pair[1]]
            imgs = [row1['fpath_img'].values[0],row2['fpath_img'].values[0]]
            segms = [ANN_ROOT_DIR + row1['fpath_segm'].values[0], ANN_ROOT_DIR + row2['fpath_segm'].values[0]]

            w, h =  row1['width'].values[0], row1['height'].values[0]

            iou = vis_pair(imgs[0], imgs[1], segms[0], segms[1])
            #if iou > 0.5 :
            new_line = {}
            new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], new_line["height"], new_line['iou'] = \
                    imgs, segms, int(w), int(h), iou
            #print(new_line)
            output += json.dumps(new_line) + "\n"


with open('../data/bodyparts_csv/train_iou.odgt', 'w') as fp:
    fp.write(output)

