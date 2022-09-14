import torch
from torch.utils.data import DataLoader
import configparser
import numpy as np
from time import time
from math import sqrt
from torch.optim.lr_scheduler import LambdaLR

from un_dataset import UN_Dataset

config=configparser.ConfigParser()
config.read('config.txt')

num_workers=int(config.get('train', 'num_workers'))
prefetch_factor=int(config.get('train', 'prefetch_factor'))
if(config.get('train', 'disable_pin_memory')=="True"):
    pin_memory=False
else:
    pin_memory=True
adam_beta1=float(config.get('train', 'adam_beta1'))
adam_beta2=float(config.get('train', 'adam_beta2'))
adam_epsilon=float(config.get('train', 'adam_epsilon'))
warmup_steps=int(config.get('train', 'warmup_steps'))
lr_factor=float(config.get('train', 'lr_factor'))
d_model=int(config.get('architecture', 'd_model'))

training_data=UN_Dataset()

dataloader = DataLoader(training_data, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)

num_of_batches=len(dataloader.dataset)


params=[]


learning_rate=lr_factor/sqrt(d_model*warmup_steps)
schedule_fn = lambda step_num: min(sqrt(warmup_steps/(step_num+1)),(step_num+1)/warmup_steps)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params, lr=learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn, verbose=False)





for epoch in range(1):
    
    s=time()
    
    for batch_num, (EN_onehot, FR_onehot, FR_int) in enumerate(dataloader):
        
        step_num=batch_num + epoch*num_of_batches
        
        print(batch_num, EN_onehot.shape, FR_onehot.shape, FR_int.shape)
        if(batch_num>0.01*num_of_batches):
            break
        
        
        
        scheduler.step()
        
    e=time()
    
print(e-s)
        
