import torch
import configparser
import sys
import argparse
import os
from tokenizers import Tokenizer
import numpy as np
from torchtext.data.metrics import bleu_score
from utils import inference_beam_search, inference_greedy_search

from model import Model
import utils

tokenizer = Tokenizer.from_file("dataset/UNv1.0/tokenizer.json")
vocab_size=tokenizer.get_vocab_size()
bos_token_id=tokenizer.token_to_id("[BOS]")
pad_token_id=tokenizer.token_to_id("[PAD]")


with open('dataset/UNv1.0/other/my_own_15_sentences.txt', 'r') as f:
    en_list= f.readlines()
    
en_inference=[]

for i in range(len(en_list)):
   
    ids_list_en = tokenizer.encode(en_list[i]).ids    
    en_inference.append(ids_list_en)
    

    
google=[]    
with open("dataset/UNv1.0/other/google_fr_translation_of_my_own_15_sentences.txt") as f:
    for line in f:
        if(line.strip()!=""):
            google.append(line)

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
beam_size=int(config.get('inference', 'beam_size'))
length_penalty=float(config.get('inference', 'length_penalty'))
inference_method=config.get('inference', 'inference_method')

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
model=Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, full_positional_encoding_matrix, masking_minus_inf, device).to(device)

loss_fn = torch.nn.CrossEntropyLoss()


checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(epoch) + '.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

references_corpus=[]
candidate_corpus=[]
candidate_corpus_no_beam=[]

print("\nPredicting ...\n")
for i in range(len(google)):
    
    print(i+1)
    print("\n")
    
    en_sentence=en_inference[i]
    
    EN=np.array(en_sentence, dtype=int)    
    EN_onehot_np=np.zeros((1,len(en_sentence),vocab_size), dtype=np.float32)
    
    for pos in range(len(en_sentence)):
        EN_onehot_np[0, pos, EN[pos]]=1

    EN_onehot=torch.tensor(EN_onehot_np).to(device)

    if(inference_method=="greedy_search"):
        fr_sentence_pred=inference_greedy_search(model, EN_onehot, device, vocab_size, bos_token_id, pad_token_id, max_len) 
    elif(inference_method=="beam_search"):
        fr_sentence_pred=inference_beam_search(model, EN_onehot, device, vocab_size, bos_token_id, pad_token_id, max_len, beam_size, length_penalty)
    else:
        sys.exit("Please, specify a valid inference_method in [inference] section of config the file. Valid values are greedy_search and beam_search.")

    french_pred=tokenizer.decode(fr_sentence_pred, skip_special_tokens = True)
    
    print(en_list[i])
    
    print(french_pred.lstrip())
    print("------------------------------------------")

    candidate_corpus.append(french_pred.split())
    references_corpus.append([google[i].split()])


score=bleu_score(candidate_corpus, references_corpus)
print("\n\n\nMy BLEU score with Google translate as reference: {0:1.4f}".format(score))





