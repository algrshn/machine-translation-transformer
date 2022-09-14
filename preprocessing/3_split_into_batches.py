# # splits the train set into batches, each batch has approximately the same
# # number of tokens (and variable number of sentences);
# # the result is two lists (one for source, and one for target) of numpy arrays,
# # with each array representing a batch

import json
import numpy as np
import configparser
import sys
from tokenizers import Tokenizer
import pickle


import utils

config=configparser.ConfigParser()
config.read('config.txt')

approx_num_of_src_tokens_in_batch=int(config.get('split_into_batches', 'approx_num_of_src_tokens_in_batch'))
approx_num_of_trg_tokens_in_batch=int(config.get('split_into_batches', 'approx_num_of_trg_tokens_in_batch'))

print("\n\nReading the dataset into RAM ...")


with open("../dataset/UNv1.0/after_step_2/en_train.json") as f:
    en_train_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_2/fr_train.json") as f:
    fr_train_input = json.load(f)

print("Done\n\n")

    
en_lengths=[]

print("Sorting the dataset ...")

for sentence in en_train_input:
    en_lengths.append(len(sentence))


zipped = sorted(zip(en_lengths,en_train_input,fr_train_input))
en_lengths, en_train_input, fr_train_input = zip(*zipped)


print("Done\n\n") 

  

print("Splitting into batches ...")


en_train_split=[]
fr_train_split=[]

en_train_batch=[]
fr_train_batch=[]

curr_num_of_sentences_in_batch=0
en_curr_max_len_in_batch=0
fr_curr_max_len_in_batch=0

en_batch_max_len=[]
fr_batch_max_len=[]

for i in range(len(en_train_input)):
    
    
    next_en_sentence=en_train_input[i]
    next_fr_sentence=fr_train_input[i]
    
    add_or_close=utils.add_another_sentence_to_batch_or_close_batch(approx_num_of_src_tokens_in_batch, approx_num_of_trg_tokens_in_batch, en_curr_max_len_in_batch, fr_curr_max_len_in_batch, curr_num_of_sentences_in_batch, next_en_sentence, next_fr_sentence)
    
    if(add_or_close=="add_to_current_batch"):
        
        en_train_batch.append(next_en_sentence)     
        fr_train_batch.append(next_fr_sentence)
        curr_num_of_sentences_in_batch=len(en_train_batch)
        en_curr_max_len_in_batch=utils.get_max_length_in_batch(en_train_batch)
        fr_curr_max_len_in_batch=utils.get_max_length_in_batch(fr_train_batch)
        
    elif(add_or_close=="close_current_batch_and_add_to_next"):
        
        en_train_split.append(en_train_batch)   # close current batch
        fr_train_split.append(fr_train_batch)
        
        en_batch_max_len.append(en_curr_max_len_in_batch)
        fr_batch_max_len.append(fr_curr_max_len_in_batch)        
        
        en_train_batch=[next_en_sentence]       # start new batch and add to it
        fr_train_batch=[next_fr_sentence]
        en_curr_max_len_in_batch=len(next_en_sentence)
        fr_curr_max_len_in_batch=len(next_fr_sentence)
        curr_num_of_sentences_in_batch=1
        
    else:
        
        sys.exit("Error! Function utils.add_another_sentence_to_batch_or_close_batch returned invalid value")

        
print("Done\n\n")



print("Converting batches to numpy arrays ...")

en_train_output=[]
fr_train_output=[]

tokenizer = Tokenizer.from_file("../dataset/UNv1.0/tokenizer.json")
pad_id=tokenizer.token_to_id("[PAD]")


for i in range(len(en_train_split)):
        
    en_train_batch=en_train_split[i]
    fr_train_batch=fr_train_split[i]
    
    en_len_max=en_batch_max_len[i]
    fr_len_max=fr_batch_max_len[i]
    
    en_train_batch_padded=[]
    fr_train_batch_padded=[]
    
    for sentence_num in range(len(en_train_batch)):
        
        en_sentence=en_train_batch[sentence_num]
        fr_sentence=fr_train_batch[sentence_num]
        
        en_len=len(en_sentence)
        fr_len=len(fr_sentence)
        
        if(en_len<en_len_max):
            en_sentence_padding=[pad_id]*(en_len_max-en_len)
            en_sentence_padded=en_sentence + en_sentence_padding
        else:
            en_sentence_padded=en_sentence
        
        if(fr_len<fr_len_max):
            fr_sentence_padding=[pad_id]*(fr_len_max-fr_len)
            fr_sentence_padded=fr_sentence + fr_sentence_padding
        else:
            fr_sentence_padded=fr_sentence 
            
        en_train_batch_padded.append(en_sentence_padded)
        fr_train_batch_padded.append(fr_sentence_padded)
    
    EN_batch=np.array(en_train_batch_padded, dtype=int)
    FR_batch=np.array(fr_train_batch_padded, dtype=int)
    
    en_train_output.append(EN_batch)
    fr_train_output.append(FR_batch)        
        

print("Done\n\n")    

print("Saving the results ...")

with open('../dataset/UNv1.0/en_train.pickle', 'wb') as f:
    pickle.dump(en_train_output, f)

with open('../dataset/UNv1.0/fr_train.pickle', 'wb') as f:
    pickle.dump(fr_train_output, f)    

print("Done")
