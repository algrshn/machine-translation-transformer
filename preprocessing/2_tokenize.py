# takes train and val sets from the folder after_step_0,
# tokenizes them and saves the results; 
# the tokenized train set files go to the folder after_step_2,
# the tokenized val set files go to the root (preprocessing/UNv1.0/) folder
# as they will not require any further preprocessing

from tokenizers import Tokenizer
import json

with open("../dataset/UNv1.0/after_step_0/en_train.json") as f:
    en_train_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_0/fr_train.json") as f:
    fr_train_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_0/en_val.json") as f:
    en_val_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_0/fr_val.json") as f:
    fr_val_input = json.load(f)
    

tokenizer = Tokenizer.from_file("../dataset/UNv1.0/tokenizer.json")

N_train=len(en_train_input)
N_val=len(en_val_input)

en_train_output=[]
fr_train_output=[]
en_val_output=[]
fr_val_output=[]

print("\nTokenizing the validation set ...")
for i in range(N_val):
    if(i % 10000 == 0):
        print(i)    
    ids_list_en = tokenizer.encode(en_val_input[i]).ids
    ids_list_fr = tokenizer.encode(fr_val_input[i]).ids
    
    en_val_output.append(ids_list_en)
    fr_val_output.append(ids_list_fr)


print("\nTokenizing the train set ...")
for i in range(N_train):
    if(i % 10000 == 0):
        print(i)    
    ids_list_en = tokenizer.encode(en_train_input[i]).ids
    ids_list_fr = tokenizer.encode(fr_train_input[i]).ids
    
    en_train_output.append(ids_list_en)
    fr_train_output.append(ids_list_fr)
    
    
with open('../dataset/UNv1.0/after_step_2/en_train.json', 'w') as f:
    json.dump(en_train_output, f)
    
with open('../dataset/UNv1.0/after_step_2/fr_train.json', 'w') as f:
    json.dump(fr_train_output, f)
    
with open('../dataset/UNv1.0/after_step_2/en_val.json', 'w') as f:
    json.dump(en_val_output, f)
    
with open('../dataset/UNv1.0/after_step_2/fr_val.json', 'w') as f:
    json.dump(fr_val_output, f)    
