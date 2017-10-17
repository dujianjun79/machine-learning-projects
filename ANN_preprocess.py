# -*- coding: utf-8 -*-
"""
Machine Learning, Assignment 3, programming part
preprocess the data, remove null, standardize, and create dummies for catorgarical variables
@author: Jianjun Du, Bo Huang
"""
import sys
import pandas as pd
import numpy as np

class preprocess:
    def __init__(self,datapath):
        self.data=pd.read_csv(datapath)        
        
    def clean(self):
        # remove all the rows with nulls
        data=self.data.dropna()
        
        # transfer all the classes labels to numbers, but not create extra dummy variables
        classes=data.iloc[:,-1]
        y=pd.Categorical(classes).codes
        y=pd.DataFrame(y)
        y.columns=['class']
        
        # process feature variables
        X=data.iloc[:,:-1]
        
        # process the numeric variables
        numeric=X.select_dtypes(include=['int64','float64'])
        normalized=(numeric-numeric.mean())/numeric.std()
        # process the categorical variables
        categorical=X.select_dtypes(include=['object'])
        if categorical.shape[1]!=0:
            dummies=pd.get_dummies(categorical,drop_first=True)
            
        # combine the processed variables
        if categorical.shape[1]!=0 and numeric.shape[1]!=0:
            X=pd.concat([normalized,dummies],axis=1)
        elif categorical.shape[1]==0:
            X=normalized
        else:
            X=dummies
            
        return X,y
    
def main():
    model=preprocess(sys.argv[1])
    X,y=model.clean()
    data=pd.concat([X,y],axis=1)
    data.to_csv(sys.argv[2],index=False)
if __name__=="__main__":
    main()
    
        
