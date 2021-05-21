from shapely.geometry import Point
from shapely.geometry import Polygon
import json
import sys
import cv2
import os
import numpy as np
import random
import pandas as pd

filename = sys.argv[1]
classes = sys.argv[2]
new_size = int(sys.argv[3])
dest_dir = sys.argv[4]

df = pd.read_csv(filename + '_fixed',  delimiter = ',', names = ['_id', 'user', 'location', 'image', 'tag', 'created', '__v'])

class_names = classes.strip(']').strip('[').split(',')
classes = {}
for ind, name in enumerate(class_names):
    classes[name] = ind + 1


df = pd.read_csv(filename + '_fixed',  delimiter = ',', names = ['_id', 'user', 'location', 'image', 'tag', 'created', '__v'])


def read_img(path):
    flag = True
    name = path.split('/')[-1]
    #path = "/home/mousavi/da1/icputrd/arf/mean.js/public/sara_img/" + name[:3] + "/" + name
    path = "/usb/seq_data_for_mit_code/" + name
    print(path)
    if os.path.isfile(path) == False:
        print("this image does not exist:" , path)
        flag = False
    img_obj = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    return img_obj, flag


def generate(df, classes, new_size, dest_dir):
    images_seen = set()
    i = 0
    for index, row in df.iterrows():
        # get the image name and object
        if i != 0:
            path = row['image']
            name = path.split('/')[-1]
            if name not in images_seen: 
                images_seen.add(name)
                img, flag = read_img(path)

                #get image size and do the resizing
                if flag:
                    height, width = img.shape[:2]
                    print(width, height)
                    if  width <= height:
                        width_percent = new_size / float(width)
                        new_height = int(float(height) * float(width_percent))
                        new_width = new_size
                        img = cv2.resize(img, (new_width, new_height))
                    elif height <= width:
                        height_percent = new_size /float(height)
                        new_width = int(float(width) * float(height_percent))
                        new_height = new_size
                        img = cv2.resize(img, (new_width, new_height)) 
                    #create an empty image
                    ann_img = np.zeros((img.shape[0],img.shape[1], 3)).astype('uint8') 
                    #ann_img = np.ones((img.shape[1],img.shape[0], 3)).astype('uint8') #create an empty image
                    height, width = ann_img.shape[:2]
                    #Find all of the tags for this image
                    df_sub = df[df['image'] == path]

                    polygons = []

                    for index2 , row2 in df_sub.iterrows():
                        location = row2['location']
                        loc = json.loads(location)
                        geometry = loc[0]['geometry'] #get the whole geomety section od the coordinate
                        geometry_points = geometry['points']# get the points of the geometry. This is a list of points(x, y) = [{}, {}, ...]
                        polygon_points = [] #this will hold the points that shape the polygon for us
                        class_id = classes[row2['tag']]
                        for p in geometry_points: # access each point to convert it from ratio to numbers 
                            x = p['x'] # x is ratio and needs to be converted to actual number
                            x = x * width
                            y = p['y']#y is ratio and needs to be converted to actual number
                            y = y * height
                            polygon_points.append((x, y))
                        polygon = Polygon(polygon_points)
                        polygons.append((class_id, polygon))


                    #for each pixel:
                    for h in range(height):
                        for w in range(width):
                            for class_id, polygon in polygons:
                                p = Point(w, h)
                                if polygon.contains(p):
                                    ann_img[h,w] = class_id #colors[class_id-1]
                                    break


                    name2 = name.replace('.JPG', ".png")
                    print("writing to {}".format(dest_dir + name2))
                    cv2.imwrite(dest_dir + name2, ann_img)
                    cv2.imwrite(dest_dir + name, img)


        i += 1


generate(df, classes, new_size, dest_dir)
