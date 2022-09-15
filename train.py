import torch
from torch.utils.data import DataLoader
import configparser
import numpy as np
from time import time
from math import sqrt
from torch.optim.lr_scheduler import LambdaLR
import sys

from un_dataset import UN_Dataset
from model import Model

config=configparser.ConfigParser()
config.read('config.txt')

device=config.get('train', 'device')
epochs=int(config.get('train', 'epochs'))
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
P_drop=float(config.get('train', 'P_drop'))
epsilon_ls=float(config.get('train', 'epsilon_ls'))

N=int(config.get('architecture', 'N'))
d_model=int(config.get('architecture', 'd_model'))
d_ff=int(config.get('architecture', 'd_ff'))
h=int(config.get('architecture', 'h'))
d_k=int(config.get('architecture', 'd_k'))
d_v=int(config.get('architecture', 'd_v'))

training_data=UN_Dataset()

dataloader = DataLoader(training_data, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)

num_of_batches=len(dataloader.dataset)
vocab_size=dataloader.dataset[0][0].shape[2]

model=Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop).cuda(device)

learning_rate=lr_factor/sqrt(d_model*warmup_steps)
schedule_fn = lambda step_num: min(sqrt(warmup_steps/(step_num+1)),(step_num+1)/warmup_steps)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn, verbose=False)


# for name, param in model.named_parameters():
#     print("{} - {}".format(name, param.shape))



for epoch in range(epochs):
    
    s=time()
    
    for batch_num, (EN_onehot, FR_onehot_shifted, FR_int) in enumerate(dataloader):
        
        step_num=batch_num + epoch*num_of_batches
        
        EN_onehot, FR_onehot_shifted, FR_int = EN_onehot.to(device), FR_onehot_shifted.to(device), FR_int.to(device)
        
        if(step_num>0.01*num_of_batches):
            sys.exit()
        
        
        FR_onehot_pred = model(EN_onehot, FR_onehot_shifted)
        
        loss=loss_fn(FR_onehot_pred, FR_int)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        print(batch_num, loss.cpu().detach().numpy())
        
    e=time()
    
    print(e-s)
        
