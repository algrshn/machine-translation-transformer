import torch
from torch import nn
import sys
import numpy as np

class ModelDense(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rates, use_batchnorm, activation, batchnorm_momentum=0.1):
        super().__init__()
                    
        if(len(hidden_dims)!=len(dropout_rates)):
            sys.exit("Cannot initialize a ModelDense model. Lengths of hidden_dims and dropout_rates are not consistent.")
                    
        layers=[]
        
        for i in range(len(hidden_dims)):
            
            if(i==0):
                prev=input_dim
            else:
                prev=hidden_dims[i-1]                        
            curr=hidden_dims[i]
            layers.append(nn.Linear(prev, curr))
            
            if(use_batchnorm):
                layers.append(nn.BatchNorm1d(curr, momentum=batchnorm_momentum))
            
            if(activation=="elu"):
                layers.append(nn.ELU())
            elif(activation=="relu"):
                layers.append(nn.ReLU())
            elif(activation=="sigmoid"):
                layers.append(nn.Sigmoid())
            else:
                sys.exit("Cannot initialize a ModelDense model. Unknown activation function was specified.")
            
            if(dropout_rates[i]!=0):
                layers.append(nn.Dropout(p=dropout_rates[i]))
        
        if(len(hidden_dims)>0):
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        
        self.dense = nn.Sequential(*layers) 
                  
                
    
    def forward(self, X):
        
        logits=self.dense(X)
                
        return logits
    
    
class RNNCell(nn.Module):
    
    def __init__(self, input_dim, output_dim, init="xavier"):
        super().__init__()
        
        if(init=="xavier"):
            self.W=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            self.T=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
        elif(init=="he"):    
            self.W=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            self.T=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
        
        self.b=nn.Parameter(torch.zeros((1,output_dim)))
        
    def forward(self, X, S):
        
        Y=torch.sigmoid(torch.matmul(X, self.W)+self.b+torch.matmul(S,self.T))
            
        return Y


class GRUCell(nn.Module):
    
    def __init__(self, input_dim, output_dim, init="xavier"):
        super().__init__()
        

        if(init=="xavier"):            
            self.W_xr=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xz=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xg=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            
            self.W_hr=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hz=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hg=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
        elif(init=="he"):
            self.W_xr=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xz=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xg=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            
            self.W_hr=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hz=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hg=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
                
        self.b_r=nn.Parameter(torch.zeros((1,output_dim)))
        self.b_z=nn.Parameter(torch.zeros((1,output_dim)))
        self.b_g=nn.Parameter(torch.zeros((1,output_dim)))
        
    def forward(self, X, H):
        
        R=torch.sigmoid(torch.matmul(X, self.W_xr)+self.b_r+torch.matmul(H,self.W_hr))
        Z=torch.sigmoid(torch.matmul(X, self.W_xz)+self.b_z+torch.matmul(H,self.W_hz))        
        G=torch.tanh(torch.matmul(X, self.W_xg)+self.b_g+torch.matmul(H*R,self.W_hg))

        Y=(torch.ones_like(Z)-Z)*G+H*Z         
            
        return Y


class LSTMCell(nn.Module):
    
    def __init__(self, input_dim, output_dim, init="xavier"):
        super().__init__()
        
        if(init=="xavier"):            
            self.W_xf=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xg=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xi=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xo=nn.Parameter(init_Xavier_tensor(input_dim=input_dim, output_dim=output_dim))
            
            self.W_hf=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hg=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hi=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_ho=nn.Parameter(init_Xavier_tensor(input_dim=output_dim, output_dim=output_dim))
        elif(init=="he"):
            self.W_xf=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xg=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xi=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            self.W_xo=nn.Parameter(init_He_tensor(input_dim=input_dim, output_dim=output_dim))
            
            self.W_hf=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hg=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_hi=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
            self.W_ho=nn.Parameter(init_He_tensor(input_dim=output_dim, output_dim=output_dim))
                
        self.b_f=nn.Parameter(torch.zeros((1,output_dim)))
        self.b_g=nn.Parameter(torch.zeros((1,output_dim)))
        self.b_i=nn.Parameter(torch.zeros((1,output_dim)))
        self.b_o=nn.Parameter(torch.zeros((1,output_dim)))
        
    def forward(self, X, C, H):
        
        F=torch.sigmoid(torch.matmul(X, self.W_xf)+self.b_f+torch.matmul(H,self.W_hf))
        I=torch.sigmoid(torch.matmul(X, self.W_xi)+self.b_i+torch.matmul(H,self.W_hi))
        O=torch.sigmoid(torch.matmul(X, self.W_xo)+self.b_o+torch.matmul(H,self.W_ho))        
        G=torch.tanh(torch.matmul(X, self.W_xg)+self.b_g+torch.matmul(H,self.W_hg))                


        C_new=G*I+F*C
        Y=O*torch.tanh(C_new)        
            
        return C_new, Y




        
        
def init_He_tensor(input_dim, output_dim):   # for ReLU
    
    W=torch.empty((input_dim, output_dim), dtype=torch.float32).normal_(std=np.sqrt(4/(input_dim+output_dim)))
    
    return W

def init_Xavier_tensor(input_dim, output_dim):  # for tanh
    
    W=torch.empty((input_dim, output_dim), dtype=torch.float32).normal_(std=np.sqrt(2/(input_dim+output_dim)))
    
    return W