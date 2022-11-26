# import torch
# from torch.utils.data import DataLoader
# import configparser
# import numpy as np
# from time import time
# from math import sqrt
# from torch.optim.lr_scheduler import LambdaLR

# from un_dataset import UN_Dataset
# from modules import ModelDense

# config=configparser.ConfigParser()
# config.read('config.txt')

# num_workers=int(config.get('train', 'num_workers'))
# prefetch_factor=int(config.get('train', 'prefetch_factor'))
# if(config.get('train', 'disable_pin_memory')=="True"):
#     pin_memory=False
# else:
#     pin_memory=True
# adam_beta1=float(config.get('train', 'adam_beta1'))
# adam_beta2=float(config.get('train', 'adam_beta2'))
# adam_epsilon=float(config.get('train', 'adam_epsilon'))
# warmup_steps=int(config.get('train', 'warmup_steps'))
# lr_factor=float(config.get('train', 'lr_factor'))
# d_model=int(config.get('architecture', 'd_model'))


# training_data=UN_Dataset()

# dataloader = DataLoader(training_data, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)

# num_of_batches=len(dataloader.dataset)


# # params=[]
# model=ModelDense(input_dim=1024, hidden_dims=[512,128,64], output_dim=10, dropout_rates=[0,0,0], use_batchnorm=False, activation='relu',)
# params=model.parameters()


# learning_rate=lr_factor/sqrt(d_model*warmup_steps)
# schedule_fn = lambda step_num: min(sqrt(warmup_steps/(step_num+1)),(step_num+1)/warmup_steps)

# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params, lr=learning_rate)
# scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn, verbose=False)





# for epoch in range(10):
    
#     s=time()
    
#     for batch_num in range(num_of_batches):
        
#         step_num=batch_num + epoch*num_of_batches
        
        
#         if(step_num % 10000 == 0):
#             print(step_num, scheduler.get_last_lr()[0])        
        
#         scheduler.step()
        

        
#     e=time()
    
# print(e-s)
        

# def myfunc(n):
#   return lambda a : a * n

# mydoubler = myfunc(2)

# print(mydoubler(11))

# import pickle

# with open('dataset/UNv1.0/en_train.pickle', 'rb') as f:
#     en_train = pickle.load(f)
    
# with open('dataset/UNv1.0/fr_train.pickle', 'rb') as f:
#     fr_train = pickle.load(f)
    
# print(fr_train[-1].shape)

# lst=[0,1,2,3,4]


# def adder(*args):
    
#     res=0
#     for arg in args:
#         res+=arg
        
#     return res


# y=adder(*lst)

# print(y)

import json

with open("dataset/UNv1.0/after_step_2/en_train.json") as f:
    en_train_input = json.load(f)
    
with open("dataset/UNv1.0/after_step_2/fr_train.json") as f:
    fr_train_input = json.load(f)
    
max_len=99
min_len=3

count_min=0
count_max=0

for i in range(len(en_train_input)):
    
    if(i % 10000 == 0):
        print(i)
    
    if(len(en_train_input[i])>max_len or len(fr_train_input[i])>max_len):
       count_max+=1
       
    if(len(en_train_input[i])<min_len or len(fr_train_input[i])<min_len):
       count_min+=1       
       
print(count_max)
print(count_min)    
    
