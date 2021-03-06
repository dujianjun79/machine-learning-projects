# -*- coding: utf-8 -*-
"""
Machine Learning, Assignment 6, part III

@author: Jianjun Du, Bo Huang
"""

import numpy as np
import cv2 
import os

img = cv2.imread('image3.jpg')
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
path=os.getcwd()
subdir='clusteredImages'
path=os.path.join(path,subdir)
final=os.path.join(path,'image3_clustered.jpg')
cv2.imwrite(final,res2)
