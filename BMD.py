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
img_set = []
for i in range(len(img_list)):
    img = hu_correction(dcm, img_list[i])
    img_set.append(img)
    seg = img * msk_img_array[:, :, len(img_list)- 1 - i]
    print(i)
    seg_list.append(seg)


slice_num = 0
plt.subplot(121)
plt.hist(img_list[slice_num].flatten(), 100, [-1024, 1024], color = 'g')
plt.xlim([-1024, 1024])
plt.legend('histogram', loc = 'upper right')
plt.subplot(122)
plt.hist(img_set[slice_num].flatten(), 100, [-1024, 1024], color = 'g')
plt.xlim([-1024, 1024])
plt.legend('histogram', loc = 'upper right')
plt.show()
print("------------Hu correction completed-------------")
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



total_area_cnt = np.count_nonzero(msk_img_array[:, :, 0] == 1)
roiimg_1, roi_cnt_1 = roi_extraction(seg_list[0], -1024, 0)


voi_cnt_1 = 0
voi_img_1 = []
for idx in range(len(seg_list)):
    voi_img, voi_cnt = roi_extraction(seg_list[idx], -1024, 0)
    voi_img_1.append(voi_img)
    voi_cnt_1 += voi_cnt
    print(idx)

total_vol_cnt = np.count_nonzero(msk_img_array == 1)

area_1 = roi_cnt_1 * float(dcm.SliceThickness) * float(dcm.PixelSpacing[0])
print("[-1024 ~ 0] volume(1slice) is " + str(area_1) + " cm^3,  " + str(round((roi_cnt_1/total_area_cnt * 100),2)) + "%")


vol_1 = voi_cnt_1 * float(dcm.SliceThickness) * float(dcm.PixelSpacing[0])
print("[-1024 ~ 0] volume is " + str(vol_1) + " cm^3,  " + str(round((voi_cnt_1/total_vol_cnt * 100),2)) + "%")




from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.figure_factory import create_trisurf
from plotly.graph_objs import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = map(float, ([scan.SliceThickness] + list(scan.PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def make_mesh(image, threshold=-300, step_size=1):

    print ("Transposing surface")
    p = image.transpose(2,1,0)
        
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    
    return verts, faces
    
def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    # Make the colormap single color since the axes are positional not intensity. 
    #colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)

def plt_3d(verts, faces):
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    print ("Drawing")
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()


imgs_after_resamp, spacing = resample(seg_list, dcm)
print ("Shape after resampling\t", imgs_after_resamp.shape)

#v, f = make_mesh(test, -350)
v, f = make_mesh(test, 350, 1)
plotly_3d(v, f)
plt_3d(v, f)