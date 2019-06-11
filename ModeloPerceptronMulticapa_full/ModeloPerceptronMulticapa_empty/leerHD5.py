# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:17:25 2018

"""

import tables
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
import h5py
import cv2


hdf5_path = 'dataset.hdf5'
subtract_mean = False
# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
# subtract the training mean
if subtract_mean:
    mm = hdf5_file["train_mean"][0, ...]
    mm = mm[np.newaxis, ...]
# Total number of samples
data_num = hdf5_file["train_img"].shape[0]

print(data_num)

#batch_size=1
#nb_class=2
#
train_img = np.uint8(hdf5_file["train_img"][0:data_num, ...])
val_img = np.uint8(hdf5_file["val_img"][0:data_num, ...])
test_img = np.uint8(hdf5_file["test_img"][0:data_num, ...])

train_labels = np.uint8(hdf5_file["train_labels"][0:data_num, ...])
val_labels = np.uint8(hdf5_file["val_labels"][0:data_num, ...])
test_labels = np.uint8(hdf5_file["test_labels"][0:data_num, ...])

cv2.imshow('imagen',train_img[0,:,:,:])
k = cv2.waitKey()
cv2.destroyAllWindows()

hdf5_file.close()