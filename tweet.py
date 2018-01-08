# -*- coding: utf-8 -*-
"""
Machine Learning, Assignment 6, Part II

@author: Jianjun Du, Bo Huang
"""

import sys
import json
import re
import numpy as np
import pandas as pd

table=[]
line_generator = open(sys.argv[1])
for line in line_generator:
    row=[]
    line_object = json.loads(line)
    text=line_object["text"]
    row.append(text)
    id_string = line_object["id"]
    row.append(id_string)
    table.append(row)

for i in range(len(table)):
    table[i][0]=re.sub('[@#]\w*','',table[i][0])
    table[i][0]=re.sub(':','',table[i][0])
    table[i][0]=re.sub('RT','',table[i][0])
    table[i][0]=re.sub('\w*//?\w*','',table[i][0])
    table[i][0]=table[i][0].split(" ")

def distance(list1, list2):
    union=set(list1).union(set(list2))
    intersection=set(list1).intersection(set(list2))
    dist=(union-intersection)/union
    return dist

cent=pd.read_csv("initial.csv",header=None)
cent=cent[0]

centroid=[]
row=[]
for item in cent:
    row.append(item)
    for element in table:
        if item==element[1]:
            row.append(element[0])
    centroid.append(row)

def k_mean(data, centroid, k):
        

            