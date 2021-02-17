# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:51:44 2021

@author: User
"""
import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os
import seaborn as sns


base_path = 'C:\\Users\\User\\Desktop\\spine\\'


#%%
#image example
images_path = base_path + 'sample'
images_list = [s for s in listdir(images_path) if isfile(join(images_path, s))]

print('Total File sizes')
for f in os.listdir(base_path):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize(base_path + '\\' + f) / 1000000, 2)) + 'MB')
        #font size 30, 왼쪽정렬

#%%
print('Number of train images:', len(images_list))

#%%
#checking images

fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 1
img_list = []
for i in range(1, columns * rows + 1):
    dcm = pydicom.dcmread(images_path + '/' + images_list[i - 1])
    fig.add_subplot(rows, columns, i)
    plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot
    img_list.append(dcm.pixel_array)

#%% dicom load
img_list = []
for i in range(len(images_list)):
    dcm = pydicom.dcmread(images_path + '\\' + images_list[i])
    img_list.append(dcm.pixel_array)
    print(i)
print("DICOM images load completed")


#%% mask load 
msk = pydicom.dcmread('C:\\Users\\User\\Desktop\\spine\\seg.dcm')
d = msk.pixel_array
msk_img = np.transpose(d, (1, 2, 0))
msk_img_array = np.array(msk_img)
print("mask images load completed")


slice_num = 50

plt.subplot(121)
plt.imshow(img_list[slice_num])
plt.subplot(122)
plt.imshow(d[103 - slice_num, :, :])
#%%
#plotting HU
def hu_correction(dcm, img):

    image = img.astype(np.int16)
    
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    
        # Convert to Hounsfield units (HU)
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
            
    image += np.int16(intercept)
    
    image[image < np.min(image)] = np.min(image)
    image[image > np.max(image)] = np.max(image)
    
    return np.array(image, dtype=np.int16)


#%% dicom * mask 

seg_list = []
for i in range(len(img_list)):
    img = hu_correction(dcm, img_list[i])
    seg = img * msk_img_array[:, :, len(img_list)- 1 - i]
    print(i)
    seg_list.append(seg)

print("------------segmentation completed--------------")
    

#%%

hist,bins = np.histogram(seg_list[0].flatten(),100,[-1000,1000])
#cdf = hist.cumsum()
#cdf_normalized = cdf * hist.max()/ cdf.max()

#plt.plot(cdf_normalized, color = 'b')
plt.hist(seg_list[0].flatten(),100,[1,1000], color = 'r')
plt.xlim([0,1000])
plt.legend('histogram', loc = 'upper left')
plt.show()



#%%

def roi_extraction(img, min_value, max_value):
    inputimg = img
    i = 0
    j = 0
    img_f = np.zeros(inputimg.shape)
    for i in range(inputimg.shape[0]):
        for j in range(inputimg.shape[1]):
            if inputimg[i, j] > int(min_value) and inputimg[i, j] < int(max_value):
                img_f[i, j] = 1
            else:
                img_f[i, j] = 0
    
    area_count = np.count_nonzero(img_f == 1)
    
    return img_f, area_count



roiimg_1, roi_cnt_1 = roi_extraction(seg_list[0], -1024, 0)


voi_cnt_1 = 0
voi_img_1 = []
for idx in range(len(seg_list)):
    voi_img, voi_cnt = roi_extraction(seg_list[idx], -1024, 0)
    voi_img_1.append(voi_img)
    voi_cnt_1 += voi_cnt
    print(idx)


area_1 = roi_cnt_1 * float(dcm.SliceThickness) * float(dcm.PixelSpacing[0])
print("-1024 ~ 0 volume(1slice) is " + str(area_1) + " cm^3")


vol_1 = voi_cnt_1 * float(dcm.SliceThickness) * float(dcm.PixelSpacing[0])
print("-1024 ~ 0 volume is " + str(vol_1) + " cm^3")


#3d 구성
https://www.kaggle.com/tamal2000/3d-display-ct-scan