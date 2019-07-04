#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:24:56 2019

@author: daniel
"""
import cv2
import os

for img_file in os.listdir('/home/daniel/Escritorio/Monitor/'):
    try:
        img = cv2.imread('/home/daniel/Escritorio/Monitor/'+img_file)
        res = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('/home/daniel/Escritorio/Monitores reescalados/' + img_file, res)
        
    except Exception as ex:
        print('ERROR------ ' + img_file)
        print(ex)
        continue