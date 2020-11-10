#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import eigh
from scipy.fftpack import fft
import glob
from scipy import linalg as la
from scipy import ndimage
import cv2
from pathlib import Path
from sklearn.metrics import f1_score

f=open("task4_mnist.txt","w")
class NN:
    
    def load_data(self,path1,nameofset):    
        data=[]
        X=[]
        imgnm= []
        rdimg = [] 
        Y=[]
        if nameofset=="Cat-Dog":
            cat = glob.glob(path1+'/cat/*.jpg')
            #print("hhhhhhhhhhh")
            for c_d in cat:
                rdimg.append((cv2.imread(c_d, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(1)
               # 	k+=1
            dog=glob.glob(path1+'/dog/dog/*.jpg') 
            #print("jjjjjj")
            for c_d in dog:  
 #               print("jjjjjjjjjjjjjjjjj")
                rdimg.append((cv2.imread(c_d, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(0)
                
        elif nameofset=="MNIST":
            i=glob.glob(path1+'/0/*.jpg')
            
            for i1 in i:
                
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(0)
                   
            i=glob.glob(path1+'/1/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(1)
            i=glob.glob(path1+'/2/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(2)
            i=glob.glob(path1+'/3/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(3)
            i=glob.glob(path1+'/4/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(4)
            i=glob.glob(path1+'/5/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(5)
            i=glob.glob(path1+'/6/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(6)
            i=glob.glob(path1+'/7/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(7)
            i=glob.glob(path1+'/8/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(8)
            i=glob.glob(path1+'/9/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(9)
            
                
  

        Y=self.one_hot_encoding(Y)
        rdimg=self.preprocessing(rdimg)
        print(np.array(rdimg).shape,Y.shape)
        # print(Y)
        return Y,rdimg


    def __init__(self,list_of_nodes):
        self.no_of_layers=len(list_of_nodes)
        self.all_layers=list_of_nodes
        self.acc=[]
        self.f1_macro=[]
        self.f1_micro=[]
    
    def one_hot_encoding(self,Y):
        encoded_list = []
        # print(Y)
        for value in Y:
            i = [0 for _ in range(10)]
            i[value] = 1
            encoded_list.append(i)
        #print(encoded_list)
        Y=np.array(encoded_list)
        return Y
    
    def preprocessing(self,rdimg):
#         rdimg1 = np.array(rdimg1)
#         mean = rdimg.mean(axis=0)
#         std = rdimg1.std(axis=0)
#         self.rdimg = (rdimg1-mean)/std
        mean,std   = np.array(rdimg).mean(axis=0),np.array(rdimg).std(axis=0)
        std = np.where(std==0,1,std)
        rdimg  = (rdimg-mean)/std
        return rdimg
    
    def softmax_Activation_function(self,net_input,r=0):
        net_input = net_input - net_input.max(axis=1,keepdims=True)
        result = np.exp(net_input)
        result = result / np.sum(result,axis=1,keepdims=True)
        if r==0:
            return result
        else:
            return result*(1-result)
        #return result / result.sum(axis=1,keepdims=True) #softmax r
    
    
    def cross_entropy(self,out_labels,y1,r=0):
        result1=(out_labels-y1)
        out_labels=np.where(y1!=1,out_labels+np.e,out_labels)
        out_labels=np.where(np.logical_and(y1==1,out_labels==0),out_labels+10**-8,out_labels)
        result= -1* np.mean(y1*np.log(out_labels),axis=0,keepdims=True)
        #print("der=0",str(result))
        #print("der=1",str(result1))
        if r==0:
            return result
        else:
            return result1
    
    def sigmoid_Activation_function(self,net_input,r=0):
        try:
            import warnings
            warnings.filterwarnings('error')
            result = 1.0 / (1.0 + np.exp(-net_input))
        except:
            print('JHGSDHDJHGDF')
            print(net_input.min())
        finally:
            warnings.filterwarnings('ignore')

        result1 = result * (1 - result)
        if r==0:
            return result
        else:
            return result1
    
     
    def relu_Activation_function(self,net_input,r=0):
        result = np.maximum(0, net_input)
        result1=(np.sign(net_input) >= 0)
        if r==0:
            return result
        else:
            return result1
    
    def swish_Activation_function(self,net_input,r=0):
       # result = net_input / (1.0 + np.exp(-net_input))
        #result1=result+ self.sigmoid_Activation_function(net_input,0) * (1-result)
        result=net_input*self.sigmoid_Activation_function(net_input,0)
        result1=self.sigmoid_Activation_function(net_input,0)
        if r==0:
            return result
        else:
            return result+result1*(1-result)
    
    def tanh_Activation_function(self,net_input,r=0):
        result=2*self.sigmoid_Activation_function(net_input,0)-1
        result1=1-(result)**2
        if r==0:
            return result
        else:
            return result1

    
    def network_init(self,activation_func,mode='gaussian'):  
        print(activation_func)
        self.activationfunc=activation_func       
        num_layers=self.all_layers
        self.weights_matrix = [0 for i in range(len(num_layers)-1)]
        self.bias=[0 for i in range(len(num_layers)-1)]  
        i=0
        for (current_layer_nodes,next_layer_nodes) in zip(num_layers[:-1],num_layers[1:]): 
            self.weights_matrix[i],self.bias[i], = self.initialize_parameters(current_layer_nodes,next_layer_nodes,self.activationfunc[i],mode)
            i+=1

#     def initWB(self,IP,OP,act_function,mode='gaussian'):
        
#         _ = 1/(IP+OP)**0.5
#         if act_function == "sigmoid" or act_function =="softmax":
#             r, s = 6**0.5, 2**0.5
#         elif act_function=='tanh':
#             r, s = 4*6**0.5, 4*2**0.5
#         elif act_function == "swish" or act_function =="relu": # relu or swish function
#             r, s = 12**0.5, 2
#         r, s = r*_, s*_
#         # Generating matrices
#         if mode=='uniform':
#             return 2*r*np.random.random((IP,OP))-r , 2*r*np.random.random((1,OP))-r
#         elif mode=='gaussian':
#             return np.random.randn(IP,OP)*s , np.random.randn(1,OP)*s
#         else:
#             raise Exception('Code should be unreachable')
            
    def initialize_parameters(self,current_layer_nodes,next_layer_nodes,act_function,state='gaussian'): #,next_layer_nodes,current_layer_nodes,act_function):
   
        k=0
        w=[]
        b=[]
#         for (current_layer_nodes,next_layer_nodes) in zip(num_layers[:-1],num_layers[1:]): 
        
        if act_function=='sigmoid' or act_function=='softmax': #or act_function=='tanh':
            if state=='gaussian':
                learning_rate=np.sqrt(2) / np.sqrt(current_layer_nodes + next_layer_nodes)
                w=(np.random.randn(current_layer_nodes,next_layer_nodes)*learning_rate)
                b=(np.random.randn(1,next_layer_nodes)*learning_rate)

            elif state=='uniform':
                learning_rate=np.sqrt(6) / np.sqrt(current_layer_nodes + next_layer_nodes)
                w=(2*learning_rate * np.random.rand(current_layer_nodes,next_layer_nodes)-learning_rate )
                b=(2*learning_rate * (np.random.rand(1,next_layer_nodes) )-learning_rate)
            else:
            	w=np.random.rand(current_layer_nodes,next_layer_nodes)
            	b=np.random.rand(1,next_layer_nodes)

        elif act_function=='relu' or act_function=='swish' :
            if state=='gaussian':
                learning_rate=2*np.sqrt(1 / (current_layer_nodes * next_layer_nodes) )
                w=(learning_rate * np.random.randn(current_layer_nodes,next_layer_nodes))
                b=(learning_rate * np.random.randn(1,next_layer_nodes))
            elif state=='uniform':
                learning_rate=np.sqrt(12 / (current_layer_nodes * next_layer_nodes) )
                w=(2*learning_rate *np.random.randn(current_layer_nodes,next_layer_nodes)-learning_rate)
                b=(2*learning_rate * np.random.randn(1,next_layer_nodes)-learning_rate)


        elif act_function=='tanh':  #or act_function=='swish':
            if state=='gaussian':
                learning_rate=4*np.sqrt(2/ (current_layer_nodes * next_layer_nodes) )
                w=(learning_rate * np.random.randn(current_layer_nodes,next_layer_nodes))
                b=(learning_rate * np.random.randn(1,next_layer_nodes))

            elif state=='uniform':
                learning_rate=4*np.sqrt(6/ (current_layer_nodes * next_layer_nodes) )
                w=(2*learning_rate *np.random.randn(current_layer_nodes,next_layer_nodes)-learning_rate)
                b=(2*learning_rate * np.random.randn(1,next_layer_nodes)-learning_rate)
        return w,b
#         self.activationfunc.append(act_function[k])
#         #print(self.weights_matrix)
#         k=k+1
 
                       
    def mini_batch(self,epochs, mini_batch_size,learning_rate): 
        training_data=self.rdimg
        Y=self.Y
        n = len(training_data)
        for j in range(epochs):
            indx  = np.arange(Y.shape[0])
            np.random.shuffle(indx)
            training_data,Y = training_data[indx], Y[indx]
            print(indx)
            sgd_batch=[]
            y1=[]
            k=0
            for l in range(int(len(training_data)/mini_batch_size)):
                sgd_batch.append(training_data[k:k+mini_batch_size])
                y1.append(Y[k:k+mini_batch_size])
                k+=mini_batch_size
            k=0
            for i in sgd_batch:
#                 print(i)
#                 input()
                result=self.forward_propagation(i,y1[k],0)
                self.backprop(y1[k],learning_rate)
                k+=1
                
    
            result=self.forward_propagation(training_data,Y,1)
            pred = 1*(result == result.max(axis=1,keepdims=True))
            print("F1- score is(Macro,Micro): ",end=' ')
            a,b=self.F1_score(pred,Y)
            print(a,b)
            self.f1_macro.append(a)
            self.f1_micro.append(b)
            
            acc1=np.mean((pred==Y).all(axis=1))
            acc1=acc1*100
            print(acc1)
            #print("jkljk")
            self.acc.append(acc1)
            f.write("EPOCH NO IS : ")
            f.write(str(j+1))
            f.write("\n-----ACCURACY-----  :::")
            f.write(str(acc1))
            f.write("\n----F1-MACRO----   :::")
            f.write(str(a))
            f.write("\n----F1-MICRO-----  :::")
            f.write(str(b))
            
            
    
            
   
    def backprop(self,y1,learning_rate):       
        change = self.cross_entropy(self.netinp_activation[-1],y1,1) * self.softmax_Activation_function(self.net_input[-1],'True')
        b_updated = change
        w_updated = np.dot(self.netinp_activation[-2].T,change)/ self.netinp_activation[-2].shape[0] 
        B = np.mean( b_updated ,axis=0, keepdims=True)
        self.weights_matrix[-1]-=learning_rate*w_updated
        self.bias[-1]-=learning_rate*B
        
        '''for l in range(self.no_of_layers-2,0,-1):
            change = np.dot(change,self.weights_matrix[l].T)*(self.sigmoid_Activation_function(self.net_input[l],1))
            b_updated= change
            w_updated = np.dot(self.netinp_activation[l-1].T,change)/ self.netinp_activation[l-1].shape[0] 
            #W = np.dot( self.IP[i].T , self.delta[i] ) / self.IP[i].shape[0] #ip[i] isthe activation of previous layer.
            B = np.mean( b_updated ,axis=0, keepdims=True) 
            self.weights_matrix[l-1]-=learning_rate*w_updated
            self.bias[l-1]-=learning_rate*B'''
        for l in range(2, self.no_of_layers):
             change = np.dot(change,self.weights_matrix[-l+1].T)* (self.sigmoid_Activation_function(self.net_input[-l],'True'))
             b_updated= change
             w_updated = np.dot(self.netinp_activation[-l-1].T,change)/ self.netinp_activation[-l-1].shape[0] 
             #W = np.dot( self.IP[i].T , self.delta[i] ) / self.IP[i].shape[0] #ip[i] isthe activation of previous layer.
             B = np.mean( b_updated ,axis=0, keepdims=True) 
             self.weights_matrix[-l]-=learning_rate*w_updated
             self.bias[-l]-=learning_rate*B

        #return b_updated,w_updated

    def F1_score(self,testlabel,predictions):
        return ((f1_score(testlabel, predictions, average='macro')),(f1_score(testlabel, predictions, average='micro')))	

   
        
    def forward_propagation(self,input_matrix,y1,v):        
        self.netinp_activation=[]       
        self.net_input=[]
        self.net_input.append(input_matrix)
        self.netinp_activation.append(input_matrix)
#        print(self.weights_matrix)
#         print(self.bias)
        for i in range(self.no_of_layers-1): 
#             print(np.dot(self.netinp_activation[i],self.weights_matrix[i]))
            result = np.dot(self.netinp_activation[i],self.weights_matrix[i])+self.bias[i]  #weights equal to the no of layers-1

#             print(self.bias[i])
#             print(self.netinp_activation[i])
#             print(result)
           
            
            
            self.net_input.append(result)           
            if self.activationfunc[i]=='sigmoid': 
#                 print("ppppp")
                output_val=self.sigmoid_Activation_function(result)
            elif self.activationfunc[i]=='softmax':
                output_val=self.softmax_Activation_function(result)
            elif self.activationfunc[i]=='tanh':
                output_val=self.tanh_Activation_function(result)
            elif self.activationfunc[i]=='swish':
                output_val=self.swish_Activation_function(result)
            elif self.activationfunc[i]=='relu':
                output_val=self.relu_Activation_function(result)
            self.netinp_activation.append(output_val)
        #print(self.netinp_activation)
        #result=self.cross_entropy(self.netinp_activation[-1],y1)
#         print(self.netinp_activation[i])
#         input()
        return self.netinp_activation[-1]
        
       

                 


# In[ ]:


cwds=os.getcwd()
path1="MNIST"
 


f1=[0 for i in range(3)]
f2=[0 for i in range(3)]
accur=[0 for i in range(3)]
k=0
list5=['sigmoid','softmax']
xaxis=[]

k=0
for i in range(3): 
    list1=[784]
    list1.append(30)
    list1.append(10)
    net = NN(list1)
    
    net.Y,net.rdimg=net.load_data(path1,"MNIST")
    
    print(np.array(net.rdimg).shape)
    if i==0:
    	net.network_init(list5)
    elif i==1:
    	net.network_init(list5,"uniform")
    elif i==2:
    	net.network_init(list5,"sig")

    
    net.mini_batch(150,30,0.01)
    f1[k]=net.f1_macro
    f2[k]=net.f1_micro
    accur[k]=net.acc
    list1=[]
    k+=1

legends=["GAUSSIAN","UNIFORM","RANDOM"]

f1=np.array(f1)
print(f1.shape)
epochs=[i for i in range(150)]

epochs=np.array(epochs)
for i in range(3):
    plt.plot(epochs,f1[i])#.reshape(1,len(f1[i])).tolist())
plt.xlabel('Epochs')
plt.ylabel('---F1-MACRO---')
plt.legend(legends)
plt.savefig(cwds+'/layer-vs-f1(macro)_task4_MNIST.png')
plt.show()
plt.clf()

for i in range(3):
    #plt.plot([1,2,3,4],[1,2,3,3])
    plt.plot(epochs,f2[i])#.reshape(1,len(f1[i])).tolist())
plt.xlabel('Epochs')
plt.ylabel('---F1-MICRO---')
plt.legend(legends)
plt.savefig(cwds+'/layer-vs-f1(micro)part1_task4_MNIST.png')
plt.show()
plt.clf()

for i in range(3):
    #plt.plot([1,2,3,4],[1,2,3,3])
    plt.plot(epochs,accur[i])#.reshape(1,len(f1[i])).tolist())
plt.xlabel('Epochs')
plt.ylabel('---ACCURACY---')
plt.legend(xaxis)
plt.savefig(cwds+'/layer-vs-acc_task4_MNIST.png')
plt.show()
plt.clf()


