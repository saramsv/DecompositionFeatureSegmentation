import cv2
import os
import numpy as np
import glob
from datetime import datetime
import json
import random


root_dir = "annotated_imgs/"
sup_train = "sup_train.odgt"
pairs = "pairs.odgt"


all_gt_files = glob.glob(root_dir + "*.png")
random.shuffle(all_gt_files)


def get_pairs(files):
	count = 0
	l = len(files)
	train_size = int(0.6 * l)
	test_size = (len(files) - train_size) // 2
	val_size = test_size
	with open(pairs, "w") as fw, open(sup_train, "w") as fwt:
		for _file in files:
			if count < train_size:
				gt = cv2.imread(_file)
				h, w = gt.shape[:2]
				img_path = _file.replace('png', 'JPG')
				first_img = [img_path]
				first_segm = [_file]
				##same_day_imgs = glob.glob( img_path.split('--')[0] + "*.png")
				count += 1
				train_line = {}
				train_line["fpath_img"], train_line["fpath_segm"], train_line["width"],\
						train_line["height"] = first_img[0] , first_segm[0] , w, h
				json.dump(train_line, fwt)
				fwt.write('\n')
				'''
				for img in same_day_imgs:
					new_line = {}
					second_img = [img]
					pair_img = first_img + second_img
					pair_segm = first_segm + first_segm # share the segmentation
					new_line["fpath_img"], new_line["fpath_segm"], new_line["width"],\
						new_line["height"] = pair_img , pair_segm , w, h
					json.dump(new_line, fw)
					fw.write('\n')
				'''
	def get_sup(files, start, end, mod):
		with open("sup_" + mod + ".odgt", "w") as fw:
			for i in range(start, end):
				new_line = {}
				fpath_img = files[i].replace('png', 'JPG') 
				fpath_segm = files[i]
				h, w = cv2.imread(fpath_img).shape[:2]
				new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], \
				new_line["height"] = fpath_img , fpath_segm , w, h
				json.dump(new_line, fw)
				fw.write('\n')
	get_sup(files, count, count + val_size, "val")
	get_sup(files, count + val_size + 1, len(files) - 1, "test")


get_pairs(all_gt_files)
