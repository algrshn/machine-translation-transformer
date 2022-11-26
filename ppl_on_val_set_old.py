import torch
import torch.nn.functional as F
import configparser
import sys
import argparse
import os
from tokenizers import Tokenizer
import json
import numpy as np
from math import log2, sqrt

from model import Model
import utils

def reduce_list(current_list):
    
    assert log2(len(current_list)).is_integer() == True
    
    reduced_list=[]
    
    for i in range(len(current_list)//2):
        reduced_list.append(sqrt(current_list[2*i]*current_list[2*i+1]))
            
    return reduced_list



tokenizer = Tokenizer.from_file("dataset/UNv1.0/tokenizer_reduced.json")
vocab_size=tokenizer.get_vocab_size()
bos_token_id=tokenizer.token_to_id("[BOS]")
pad_token_id=tokenizer.token_to_id("[PAD]")

with open("dataset/UNv1.0/en_val_reduced.json") as f:
    en_inference = json.load(f)
    
with open("dataset/UNv1.0/fr_val_reduced.json") as f:
    fr_inference = json.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default="config.txt")
args = parser.parse_args()

run_config_file=args.conf

if(run_config_file[-4:]!=".txt"):
    sys.exit("--conf must be a .txt file")

config=configparser.ConfigParser()
config.read(os.path.join("run_configs",run_config_file))

device=config.get('inference', 'device')
epoch=int(config.get('inference', 'epoch'))
max_len=int(config.get('inference', 'max_len'))
sentence_group_size_for_calculating_perplexity=int(config.get('inference', 'sentence_group_size_for_calculating_perplexity'))

if(log2(sentence_group_size_for_calculating_perplexity).is_integer()==False):
    sys.exit("The parameter sentence_group_size_for_calculating_perplexity in [inference] section of config file must be set to a power of 2. Exiting now.")


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

P_drop=0

full_positional_encoding_matrix=torch.as_tensor(utils.prepare_full_positional_encoding_matrix(d_model, positional_encoding_wavelength_scale, positional_encoding_max_pos), dtype=torch.float32, device=device)
model=Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, full_positional_encoding_matrix, masking_minus_inf, device).cuda(device)

loss_fn = torch.nn.CrossEntropyLoss()


checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(epoch) + '.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

sentence_perplexities=[]
print("\nGoing through validation set ...\n")
for i in range(len(en_inference)):
    
    utils.display_progress(i, len(en_inference))
        
    en_sentence=en_inference[i]
    fr_sentence=fr_inference[i]
        
    EN=np.array(en_sentence, dtype=int)
    FR=np.array(fr_sentence, dtype=int)
    
    FR_shifted=np.zeros_like(FR)
    FR_shifted[1:]=FR[:-1]
    FR_shifted[0]=bos_token_id
    
    
    EN_onehot_np=np.zeros((1,len(en_sentence),vocab_size), dtype=np.float32)
    FR_onehot_shifted_np=np.zeros((1,len(fr_sentence),vocab_size), dtype=np.float32)
    
    FR_onehot_shifted_np[0,0,bos_token_id]=1
    
    for pos in range(len(en_sentence)):
        EN_onehot_np[0, pos, EN[pos]]=1
    for pos in range(len(fr_sentence)):
        FR_onehot_shifted_np[0, pos, FR_shifted[pos]]=1

    EN_onehot=torch.tensor(EN_onehot_np).to(device)
    FR_onehot_shifted=torch.tensor(FR_onehot_shifted_np).to(device)

    FR_onehot_pred = model(EN_onehot, FR_onehot_shifted)
    FR_onehot_prob=F.softmax(FR_onehot_pred, dim=1).cpu().detach().numpy()
    
    perplexity_not_normalized=1
    for pos in range(len(fr_sentence)):
        word_probability=FR_onehot_prob[0,fr_sentence[pos],pos]
        perplexity_not_normalized=perplexity_not_normalized/word_probability
    
    sentence_perplexity=perplexity_not_normalized**(1/len(fr_sentence))    
    sentence_perplexities.append(sentence_perplexity)   

num_of_groups=len(en_inference)//sentence_group_size_for_calculating_perplexity

group_perplexities=[]
for i in range(num_of_groups):
    sentence_perplexities_within_group=sentence_perplexities[i*sentence_group_size_for_calculating_perplexity:(i+1)*sentence_group_size_for_calculating_perplexity]
    
    while 1:
        if(len(sentence_perplexities_within_group)>1):
            sentence_perplexities_within_group=reduce_list(sentence_perplexities_within_group)
        else:
            group_perplexity=sentence_perplexities_within_group[0]
            break
    
   
    group_perplexities.append(group_perplexity)
    

total_perplexity_not_normalized=1
for i in range(len(group_perplexities)):
    total_perplexity_not_normalized=group_perplexities[i]*total_perplexity_not_normalized
    
total_perplexity=total_perplexity_not_normalized**(1/len(group_perplexities))

print("\n\n\nValidation set perpexity: {0:4.2f}".format(total_perplexity))
    
    
    
    


    

