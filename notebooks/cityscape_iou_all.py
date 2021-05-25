import os
import cv2
import math
import json
import glob
import random
import argparse
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import namedtuple
from matplotlib import pyplot as plt

# helper functions
def vis_pair(img1, img2, label1, label2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    label1 = cv2.imread(label1)
    label2 = cv2.imread(label2)
    
    tmp_label2 = cv2.resize(label2, (int(label1.shape[1]), int(label1.shape[0])))
    
    label1[np.where(label1 == 0)] = 255 # to exclude the bg from intersection
    
    intersection = np.where(label1[:,:,0] == tmp_label2[:,:,0])[0].shape[0]
    union = np.where(label1[:,:,0] != 255 )[0].shape[0] + np.where(tmp_label2[:,:,0] != 0 )[0].shape[0] - intersection

    return intersection/union

def class_pair_iou(df, CLASS_LIST):
    output = ""
    for i, c in enumerate(CLASS_LIST):
        if c != 'unlabeled':
            sub_data = df[df[c]==1]
            count = 0
            pbar = tqdm(total=sum(1 for ignore in itertools.combinations(sub_data['image'], 2)))
            for pair in itertools.combinations(sub_data['image'], 2):
                pbar.set_description("{}".format(c))
                pbar.update(1)
                row1 = df[df['image']==pair[0]]
                row2 = df[df['image']==pair[1]]
                imgs = [row1['image'].values[0],row2['image'].values[0]]
                segms = [row1['segmentation'].values[0], row2['segmentation'].values[0]]
                w, h =  row1['width'].values[0], row1['height'].values[0]
                iou = vis_pair(imgs[0], imgs[1], segms[0], segms[1])
                new_line = {}
                new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], new_line["height"], new_line['iou'] = \
                        imgs, segms, int(w), int(h), iou
                output += json.dumps(new_line) + "\n"
                # break
            pbar.close()
    return output

def save_odgt(file_to_save, output_path, overwrite=False):
    if os.path.exists(file_to_save) and not overwrite:
        print("File exists!")
        return
    elif os.path.exists(file_to_save) and overwrite:
        print("Overwrite file!")
    with open(output_path, 'w') as fp:
        fp.write(file_to_save)
        
def plot_hist_from_odgt(file_path, greater_than=0.5):
    df_pairs = pd.read_json(file_path, lines=True)
    df_pairs[df_pairs['iou'] > greater_than]['iou'].hist()


    
# Global variables
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


# modified 'license plate' from -1 -> 34 and 19
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , 34 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

CLASS_MAP = {val.id:val.name for val in labels}
COLOR_MAP = {val.id:val.color for val in labels}

EVAL_CLASS_MAP = {}
EVAL_COLOR_MAP = {}

for val in labels:
    if not val.ignoreInEval:
        EVAL_CLASS_MAP[val.trainId] = val.name
        EVAL_COLOR_MAP[val.trainId] = val.color
        
CLASS_LIST = list(EVAL_CLASS_MAP.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sequence odgt file for cityscape eval train data")
    parser.add_argument("--csv", default="../data/cityscape_csv/train.csv", type=str, metavar='')
    parser.add_argument("-o", "--overwrite", default=True, type=str, metavar='', help="overwrite")
    parser.add_argument("-s", "--save", default="../data/cityscape_csv/eval_train_all_iou.odgt", type=str, metavar='', help="overwrite")
    args = parser.parse_args()
    
    print("\nread: {}".format(args.csv))
    df = pd.read_csv(args.csv)
    
    print()
    output = class_pair_iou(df, CLASS_LIST)
    
    print("\noutput: {}\n".format(args.save))
    save_odgt(output, args.save, args.overwrite)