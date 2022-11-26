# removes sentence pairs where at least one sentence (src or trg) is too long (more than max_len) or too short (less than min_len) 

import json
import configparser

config=configparser.ConfigParser()
config.read('config.txt')

max_len=int(config.get('remove_too_long_too_short', 'max_len'))
min_len=int(config.get('remove_too_long_too_short', 'min_len'))


print("\n\nReading the dataset into RAM ...")

with open("../dataset/UNv1.0/after_step_2/en_train.json") as f:
    en_train_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_2/fr_train.json") as f:
    fr_train_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_2/en_val.json") as f:
    en_val_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_2/fr_val.json") as f:
    fr_val_input = json.load(f)

print("Done")


print("\n\nProcessing validation set ...")    
en_val_output=[]
fr_val_output=[]
for i in range(len(en_val_input)):
    if(len(en_val_input[i])>=min_len and len(fr_val_input[i])>=min_len and len(en_val_input[i])<=max_len and len(fr_val_input[i])<=max_len):
        en_val_output.append(en_val_input[i])
        fr_val_output.append(fr_val_input[i])
print("Done")        

print("\n\nProcessing train set ...")
en_train_output=[]
fr_train_output=[]
for i in range(len(en_train_input)):
    if(len(en_train_input[i])>=min_len and len(fr_train_input[i])>=min_len and len(en_train_input[i])<=max_len and len(fr_train_input[i])<=max_len):
        en_train_output.append(en_train_input[i])
        fr_train_output.append(fr_train_input[i])
print("Done") 

print("\n\nSaving the results ...")        
with open('../dataset/UNv1.0/after_step_3/en_train.json', 'w') as f:
    json.dump(en_train_output, f)
    
with open('../dataset/UNv1.0/after_step_3/fr_train.json', 'w') as f:
    json.dump(fr_train_output, f)
    
with open('../dataset/UNv1.0/after_step_3/en_val.json', 'w') as f:
    json.dump(en_val_output, f)
    
with open('../dataset/UNv1.0/after_step_3/fr_val.json', 'w') as f:
    json.dump(fr_val_output, f)
print("Done")    