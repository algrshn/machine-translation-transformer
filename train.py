import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import configparser
from time import time
from math import sqrt
from torch.optim.lr_scheduler import LambdaLR
import sys
import argparse
import os

from un_dataset import UN_Dataset
from model import Model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default="config.txt")
args = parser.parse_args()

run_config_file=args.conf

if(run_config_file[-4:]!=".txt"):
    sys.exit("--conf must be a .txt file")

config=configparser.ConfigParser()
config.read(os.path.join("run_configs",run_config_file))

device=config.get('train', 'device')
epochs=int(config.get('train', 'epochs'))
resume_training_starting_with_epoch=int(config.get('train', 'resume_training_starting_with_epoch'))
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
positional_encoding_wavelength_scale=float(config.get('train', 'positional_encoding_wavelength_scale'))
masking_minus_inf=float(config.get('train', 'masking_minus_inf'))
folder_to_save_state_dicts=config.get('train', 'folder_to_save_state_dicts')

N=int(config.get('architecture', 'N'))
d_model=int(config.get('architecture', 'd_model'))
d_ff=int(config.get('architecture', 'd_ff'))
h=int(config.get('architecture', 'h'))
d_k=int(config.get('architecture', 'd_k'))
d_v=int(config.get('architecture', 'd_v'))
positional_encoding_max_pos=int(config.get('architecture', 'positional_encoding_max_pos'))

tensorboard_folder_name=run_config_file[:-4]
writer = SummaryWriter('tensorboard/' + tensorboard_folder_name)

folder_exists = os.path.isdir(folder_to_save_state_dicts)
if(not folder_exists):
    os.makedirs(folder_to_save_state_dicts)

training_data=UN_Dataset()

dataloader = DataLoader(training_data, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)

num_of_batches=len(dataloader.dataset)
vocab_size=dataloader.dataset[0][0].shape[2]

full_positional_encoding_matrix=torch.as_tensor(utils.prepare_full_positional_encoding_matrix(d_model, positional_encoding_wavelength_scale, positional_encoding_max_pos), dtype=torch.float32, device=device)
model=Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, full_positional_encoding_matrix, masking_minus_inf, device).cuda(device)

learning_rate=lr_factor/sqrt(d_model*warmup_steps)
schedule_fn = lambda step_num: min(sqrt(warmup_steps/(step_num+1)),(step_num+1)/warmup_steps)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=epsilon_ls)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn, verbose=False)

if(resume_training_starting_with_epoch!=0):
    checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(resume_training_starting_with_epoch-1) + '.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


for epoch in range(resume_training_starting_with_epoch, epochs):
    
    s=time()
    
    print(f"\n\n\nEpoch {epoch}\n")
    
    total_loss=0
    for batch_num, (EN_onehot, FR_onehot_shifted, FR_int) in enumerate(dataloader):
        
        utils.display_progress(batch_num, num_of_batches)
                       
        EN_onehot, FR_onehot_shifted, FR_int = EN_onehot.to(device), FR_onehot_shifted.to(device), FR_int.to(device)
        
        FR_onehot_pred = model(EN_onehot, FR_onehot_shifted)
        
        loss=loss_fn(FR_onehot_pred, FR_int)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        total_loss+=loss.cpu().detach().numpy()
    
    
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    }, folder_to_save_state_dicts + '/state_dict_e' + str(epoch) + '.pt')
        
    e=time()
    
    print("\nEpoch running time: {0:4.1f} hours".format((e-s)/3600))
    print("Loss: {0:4.4f}".format(total_loss/num_of_batches))
    
    writer.add_scalar("Loss/train", total_loss/num_of_batches, epoch)
    writer.flush()
        
writer.close()