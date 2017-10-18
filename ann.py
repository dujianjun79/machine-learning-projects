# -*- coding: utf-8 -*-
"""
Machine Learning, Assignment 3, coding part

@authors: Jianjun Du, Bo Huang
"""
import numpy as np
import pandas as pd
import random

class ANN:
    def __init__(self,data, percent, layers, neuros):
        #input data in pansas dataframe, specify the path and name, string
        self.data=pd.read_csv(data,header=0,dtype=np.float32) 
        self.percent=percent #percentage of data for training
        self.layers=layers # how many hidden layers in ANN
        # specify the neuros in each hidden layer
        # example, 3 layers, and 3,4,5 neuros for each layers counting from the biginning
        # the code should be [3,4,5]
        self.neuros=neuros
        # records is the number of rows
        records=self.data.shape[0]
        # split is the number of the rows distributed to train dataset
        split=int(records*percent)
        # randomly choose row numbers from the row index
        train_row=random.sample(range(records),split)
        self.train=self.data.iloc[train_row,:]
        # the test row index is the difference between that of original and train
        test_row=list(set(range(150)).difference(set(train_row)))
        self.test=self.data.iloc[test_row,:]
        
    def fit(self,learning_rate,epoches,reg):
        #initialize the parameters
        weights,bias=self.initializeWeights()
        a=self.initializeAcivation()
        # error has the same data structure with activation
        error=self.initializeAcivation()
        # maximumly run 500 epoches
        for i in range(epoches):
            # randomly select 50 samples to train the algrithm, which is called schomatic gradient descent
            # a0 is the input; T is the output, which has been encode to matrix
            samples,T=self.randomSelect(50)
            deltaW,deltab=self.initializeDelta()
            for n in range(50):
                a[0]=samples[n]
                # forward propogation
                for i in range(len(weights)):
                    z=np.dot(a[i],weights[i])+bias[i]
                    # output of the activation function, here is sigmoid
                    a[i+1]= 1/(1+np.exp(-z))    
                # this is the output layer, and the activation function is softmax
                a[-1]=self.softmax(a[-2],weights[-1],bias[-1])
                # backward propogation
                error[-1]=a[-1]-T[n]
                for i in range(-1,-len(a),-1):
                    tmp1=np.multiply(a[i-1],(1-a[i-1]))
                    tmp2=np.dot(weights[i],error[i])
                    error[i-1]=np.multiply(tmp1,tmp2)
                    tmp3=np.reshape(a[i-1],(-1,1))
                    tmp4=np.reshape(error[i],(1,-1))
                    deltaW[i]=deltaW[i]+learning_rate*np.matmul(tmp3,tmp4)
                    deltab[i]=deltab[i]+learning_rate*error[i]
            deltaW=[element/50 for element in deltaW]
            deltab=[element/50 for element in deltab]
            for i in range(len(deltaW)):
                weights[i]=weights[i]-deltaW[i]-learning_rate*reg*weights[i]
                bias[i]=bias[i]-deltab[i]-learning_rate*reg*deltab[i]
        return weights,bias
    
    def predict(self,weights,bias):
        X,y=self.test.iloc[:,:-1],self.train.iloc[:,-1]
        target=pd.get_dummies(y).values
        data=X.values
        records=data.shape[0]
        count=0
        a=self.initializeAcivation()
        for n in range(len(data)):
                a[0]=data[n]
                for i in range(len(weights)):
                    z=np.dot(a[i],weights[i])+bias[i]
                    a[i+1]= 1/(1+np.exp(-z))    
                a[-1]=self.softmax(a[-2],weights[-1],bias[-1])
                max=0
                for i in range(len(a[-1])):
                    if a[-1][i]>a[-1][max]:
                        max=i
                if target[n][i]==1:
                    count +=1
        return float(count/records)
                
    def initializeWeights(self):
        #change the dataframe to numpy array
        data=self.data.values
        # Number of feature variables, exclude the classes
        features=self.train.shape[1]-1
        #initialize the parameters
        weights=[]
        bias=[]
        for neuro in self.neuros:
            # The shape of the weights is a matrix that column is the number of  neuro, and row is the number of features
            weights.append(np.random.uniform(-1,1,size=features*neuro).reshape((features,neuro)))
            # The shape of the bias is a vector of the number of neuros
            bias.append(np.random.uniform(-1,1,size=neuro))
            # the input to next level is the output of this level
            features=neuro
        # the number of output neuros
        output_neuro=np.unique(data[:,-1]).shape[0]
        # initialize the W and b of the output neuros
        weights.append(np.random.uniform(-1,1,size=features*output_neuro).reshape(features,output_neuro))
        bias.append(np.random.uniform(-1,1,size=output_neuro))
        return weights,bias
    

    def initializeDelta(self):
        data=self.train.values
        features=self.train.shape[1]-1
        deltaW=[]
        deltab=[]
        for neuro in self.neuros:
            deltaW.append(np.zeros(features*neuro).reshape((features,neuro)))
            deltab.append(np.zeros(neuro))
            features=neuro
        output_neuro=np.unique(data[:,-1]).shape[0]
        deltaW.append(np.zeros(features*output_neuro).reshape(features,output_neuro))
        deltab.append(np.zeros(output_neuro))
        return deltaW,deltab

    def initializeAcivation(self):
        data=self.train.values
        # a store the input to each layer
        # initialize them to zeros
        output_neuro=np.unique(data[:,-1]).shape[0]
        a_dims=[self.train.shape[1]-1]
        for element in self.neuros:
            a_dims.append(element)
        a_dims.append(output_neuro)
        a=[]
        for n in a_dims:
            a.append([0]*n)
        return a
    
    # randomly select 50 samples for parameter updating, which is called SGD    
    def randomSelect(self,num):
        X,y=self.train.iloc[:,:-1],self.train.iloc[:,-1]
        y=pd.get_dummies(y)
        row=random.sample(range(self.train.shape[0]),num)
        # X is the feature matrix, and y is the response
        X,y=X.iloc[row,:],y.iloc[row,:]
        # for classification application, the response needs to transfor to a matrix
        return X.values, y.values
        
    def softmax(self,H,Wo,bo):
        z=np.dot(H,Wo)+bo
        return np.exp(z)/np.sum(np.exp(z))
            
def main():
    model=ANN('iris1.csv',0.8,2,[10,5])
    a,T=model.randomSelect(50)
    weights,bias=model.fit(0.05,50,0.2)
    result=model.predict(weights,bias)
    print("test accuracy: "+str(result))
    
if __name__=="__main__":
    main()
        
