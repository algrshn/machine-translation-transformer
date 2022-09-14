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
        

def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))

