#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sahi 
@File    ：slicePredict.py
@Author  ：kuisu
@Email     ：kuisu_dgut@163.com
@Date    ：2022/3/23 22:13 
'''
import numpy as np

from sahi.predict import get_sliced_prediction
from sahi.model import DetectionModelUNet
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# init UNet model
detection_model = DetectionModelUNet(model_path="checkpoints/checkpoint_epoch20.pth",
                                     category_mapping={"0":"0","1":"1"})
filename = r"E:\kuisu\Datasets\BGI_EXAM\test_set\171.jpg"
# filename = r"data/test/img1_6560_14145_6816_14401.jpg"
save_result_path = r"test_result_1792x1792.jpg"
# filename = r"E:\kuisu\Datasets\BGI_EXAM\slices\mask\img1_5740_12710_5996_12966.jpg"
big_image = Image.open(filename)
big_image = big_image.convert("RGB")

# get sliced prediction result

#按行分割图片
def cut_image(image,size):
    width, height = image.size
    count_w,count_h = width/size,height/size
    count = int(min(count_h,count_w))
    print("image cut to {}*{}".format(count,count))
    item_width = size
    item_height = size
    # assert item_height>=size
    # assert item_width>=size
    image_list = []
    # (left, upper, right, lower)
    for i in range(0,count):
        column = []
        for j in range(0,count):
            box = (j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height)
            column.append(box)
        row_image = [image.crop(box) for box in column]
        image_list.append(row_image)
    return image_list

# 按行拼接图片
def merge_image(image_list):
    target_width,target_height = 0,0
    for column_image in image_list[0]:
        target_width += column_image.size[0]
    for row_image in image_list:
        target_height += row_image[0].size[1]


    target = Image.new("RGB",(target_width,target_height))
    for i, row_images in enumerate(image_list):
        for j,column_images in enumerate(row_images):
            image = column_images
            item_width,item_height = image.size
            target.paste(image,(j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height))
    return target

image_list = cut_image(big_image, size=1536)

target_list = []
for row_images in image_list:
    target_column = []
    for column_images in row_images:
        image = column_images
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height = 256,
            slice_width = 256,
            overlap_height_ratio = 0.0,
            overlap_width_ratio = 0.0
        )
        if len(result.object_prediction_list)<=1:
            print('None')
            w,h = image.size
            mask_result = np.zeros((h,w))
        else:
            mask_result = result.object_prediction_list[1].mask.bool_mask
        mask_result = Image.fromarray(mask_result)
        target_column.append(mask_result)
    target_list.append(target_column)
target_result = merge_image(target_list)
target_result.save(save_result_path,quality=100)