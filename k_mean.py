# -*- coding: utf-8 -*-
"""
Machine Learning, Assignment 6, Part I

@author: Jianjun Du, Bo Huang

"""
import  sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# read in the data

data=pd.read_csv(sys.argv[1],sep='\t')
data=data.iloc[:,1:].values
# have a scatter diagram
plt.scatter(data[:,0], data[:,1], c='black', s=7)

def k_mean(data,k):
    # randomlhy choose three data points as centroid
    np.random.shuffle(data)
    centroid=data[:k,:]
    # add another column to indicate the cluster it belongs to
    z=np.zeros((data.shape[0],1))
    data=np.append(data,z,axis=1)
    sse=0
    
    while True:
        sse=0
        for i in range(data.shape[0]):
            minimum=sys.maxsize
            for j in range(centroid.shape[0]):
                distance=np.sqrt(np.sum((data[i,:-1]-centroid[j,:])**2))
                if minimum>distance:
                    minimum=distance
                    data[i][2]=j
                sse +=minimum
        data1=pd.DataFrame(data)
        data1[2]=data1[2].astype("int64")
        grouped=data1.groupby(by=data1[2],axis=0)
        new_centroid=grouped.sum()/grouped.count()
        new_centroid=new_centroid.values
        if np.sum((new_centroid-centroid)**2)<1e-3:
            break
        else:
            centroid=new_centroid
    return data,centroid,sse    

for k in range(5):
    mydata,centroid,sse=k_mean(data,3+k)
    result=pd.DataFrame(mydata)
    result[2]=result[2].astype("int64")
    result.to_csv(sys.argv[2])
    print("The SSE is "+str(sse))
    
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k+3):
            points = np.array([mydata[j,:-1] for j in \
                               range(mydata.shape[0]) if mydata[j,-1] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i-3])
    ax.scatter(centroid[:,0], centroid[:,1], marker='*', s=200, c='g')
              