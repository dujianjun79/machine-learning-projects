# -*- coding: utf-8 -*-
"""
Decision tree with only binary variables, and binary classes
@author: Jianjun Du, Bo Huang
"""
"""
Reference:
Machine Learning, Tom Mitchell, Mcgraw-hill, 1997
Machine Learing in Action, Peter harrington, Manning Publications, 2012
"""
import sys
import numpy as np
import pandas as pd
from math import log
import random

class TreeNode(object):
    def __init__(self, key, data, predict=-1):
        # key store the column names
        self.key = key
        # data store the sub dataset used for splitting
        self.data = data
        # predict stores the classes it belongs to if it is a leaf node
        self.predict=predict
        self.left = None
        self.right = None
        
class DecisionTree():        
    
    def __init__(self):
        self.node = None
    #   calculate the entropy of a dataset
    def calcEntropy(self,data):
        rowNum=len(data)
        count0=0
        for featVec in data:
            if featVec[-1]==0:
                count0 +=1
        count1=rowNum-count0
        if count0==0 or count1==0:
            return 0
        Entropy=0.0
        prob0=float(count0)/rowNum
        prob1=float(count1)/rowNum
        Entropy=-prob0*log(prob0,2)-prob1*log(prob1,2)
        return Entropy

    #   find the subdata by feature and label
    def splitData(self,data, feature, value):
        mask=np.where(data[:,feature]==value)
        return data[mask,:][0]
    
    
 
    #   find the best feature to split the dataset    
    def chooseFeature(self,data):
        numFeatures=data.shape[1]-1
        baseEntropy=self.calcEntropy(data)
        maxInfoGain=0.0
        bestFeature=-1
        for i in range(numFeatures):
            subData0=self.splitData(data,i,0)
            subData1=self.splitData(data,i,1)
            prob0=subData0.shape[0]/data.shape[0]
            prob1=subData1.shape[0]/data.shape[0]
            newEntropy=prob0*self.calcEntropy(subData0)+prob1*self.calcEntropy(subData1)
            infoGain=baseEntropy-newEntropy
            if(infoGain>maxInfoGain):
                maxInfoGain=infoGain
                bestFeature=i
        return bestFeature

    #   This function is used to find the majority vote of a split
    def voteClass(self,classVector):
        count1=np.count_nonzero(classVector)
        count0=len(classVector)-count1
        if count0>count1:
            return 0
        else:
            return 1
    
    #   This function is used to build a tree
    #   Splitting criterion is stored in a tree    
    def buildTree(self,data,label):
        #   if the data is empty, stop splitting
        if data.shape[0]==0:
            return None
        #   extract the count of each class of the data
        classVector=data[:,-1]
        #   when all the features are splitted, stop splitting
        if data.shape[1]==1:
            return TreeNode(None,data,self.voteClass(classVector))
        #   when the class is pure, stop splitting
        if np.count_nonzero(classVector)==0 or np.count_nonzero(classVector)==classVector.size:
            return TreeNode(None,data,classVector[0])
    
        Feat=self.chooseFeature(data)
        Label=label[Feat]
        #   Copy the element of label, otherwise it will be a reference
        sublabel=label[:]
        tree=TreeNode(Label,data)
        #   reduce the feature pool for the next level    
        del(sublabel[Feat])
        #   find the sub dataset for the left sub-tree and right sub-tree
        left=np.delete(self.splitData(data,Feat,0),Feat,1)
        right=np.delete(self.splitData(data,Feat,1),Feat,1)
        #   recursively split the data. Left is always the sub dataset when the feature is 0    
        tree.left=self.buildTree(left,sublabel)
        tree.right=self.buildTree(right,sublabel)
        return tree

    def predict_helper(self,data, lookup, root):
        if root==None:
            return 0
        if root.predict!=-1:
            return root.predict
        else:
            key=root.key
            index=lookup[key]
            if data[index]==0:
                return self.predict_helper(data,lookup,root.left)
            else:
                return self.predict_helper(data,lookup,root.right)
        
    def predict(self,data, lookup, root):
        predicted=[]
        actual=data[:,-1].tolist()
        for i in range(data.shape[0]):
            tmp=self.predict_helper(data[i],lookup,root)
            predicted.append(tmp)
        count=0
        for i in range(data.shape[0]):
            if predicted[i]==actual[i]:
                count +=1
        return float(count/data.shape[0])
    
    # This function is used to store all the nodes in a list
    def prunne_helper(self,root,nodelist):
        if root==None:
            return
        nodelist.append(root)
        self.prunne_helper(root.left,nodelist)
        self.prunne_helper(root.right,nodelist)
    
    # randomly generate the nodes which will be prunned.
    def prunne(self,root):
        nodelist=[]
        self.prunne_helper(root,nodelist)
        population=len(nodelist)
        prunned_number=int(0.2*population)
        seq=list(range(population))
        prunned_index=random.sample(seq,prunned_number)
        # Every prunned node becomes the leaf node, and the classes it assigned will be generated.
        for i in prunned_index:
            nodelist[i].left=None
            nodelist[i].right==None
            classVector=nodelist[i].data[:,-1]
            nodelist[i].predict=self.voteClass(classVector)
        
    def leafnode(self,root):
        if root==None:
            return 0
        if root.left==None and root.right==None:
            return 1
        else:
            return self.leafnode(root.left)+self.leafnode(root.right)
        
    def printtree(self,root, i):
        if root==None:
            return
        if root.left==None and root.right==None:
            print(i*"|"+"belong to: "+str(root.predict)+"}")
        else:
            print(i*"|"+root.key+"  as 0 : {")
            self.printtree(root.left,i+1)
            print(i*"|"+root.key+"  as 1 : {")
            self.printtree(root.right,i+1)
            
def main():
    #   read the data in using pandas
    train=pd.read_csv(sys.argv[1])
    test=pd.read_csv(sys.argv[2])
    validation=pd.read_csv(sys.argv[3])
    #   extract the column names from the dataset    
    label=list(train.columns)
    #   transfer the data from dataframe to numpy narray        
    trainData=train.values
    valData=validation.values
    testData=test.values
    #   lookup map between column name and column index    
    lookup={}
    for index, item in enumerate(label):
        lookup[item]=index
        
    model=DecisionTree()
    root=model.buildTree(trainData,label)
    nodelist=[]
    model.prunne_helper(root,nodelist)
    result=model.predict(trainData,lookup,root)
    leafnode_number=model.leafnode(root)
    model.printtree(root,1)
    print("--------------------------------------------")
    print("Pre-prunned accuracy")
    print("--------------------------------------------")
    print("Number of trainning instances are "+str(trainData.shape[0]))
    print("Number of trainning attributes are "+str(trainData.shape[1]))
    print("Total number of nodes in the tree "+str(len(nodelist)))
    print("Total number of leaf nodes are "+str(leafnode_number))
    print("Accuracy of model on trainning data is "+str(result))
    print("")
    result=model.predict(valData,lookup,root)
    print("Number of validation instances are "+str(valData.shape[0]))
    print("Number of validation attributes are "+str(valData.shape[1]))
    print("Accuracy of model on validation before prunning is "+str(result))
    print("")
    result=model.predict(testData,lookup,root)
    print("Number of test instances are "+str(testData.shape[0]))
    print("Number of test attributes are "+str(testData.shape[1]))
    print("Accuracy of model on test before prunning is "+str(result))
    print("")    
    
    model.prunne(root)
    nodelist=[]
    model.prunne_helper(root,nodelist)
    result=model.predict(trainData,lookup,root)
    leafnode_number=model.leafnode(root)
    print("post-prunned accuracy")
    print("--------------------------------------------")
    print("Number of trainning instances are "+str(trainData.shape[0]))
    print("Number of trainning attributes are "+str(trainData.shape[1]))
    print("Total number of nodes in the tree "+str(len(nodelist)))
    print("Total number of leaf nodes are "+str(leafnode_number))
    print("Accuracy of model on trainning data is "+str(result))
    print("")
    result=model.predict(valData,lookup,root)
    print("Number of validation instances are "+str(valData.shape[0]))
    print("Number of validation attributes are "+str(valData.shape[1]))
    print("Accuracy of model on validation after prunning is "+str(result))
    print("")
    result=model.predict(testData,lookup,root)
    print("Number of test instances are "+str(testData.shape[0]))
    print("Number of test attributes are "+str(testData.shape[1]))
    print("Accuracy of model on test after prunning is "+str(result))
    print("")  
    
if __name__=="__main__":
    main()

