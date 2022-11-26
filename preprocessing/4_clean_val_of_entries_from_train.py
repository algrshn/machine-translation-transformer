# The original data may contain duplicates. If one of these duplicates ends up in train set and another ends up in val set, then we get a contaminated val set. This script removes entries from val set which are also present in the train set.

import json

import utils


print("\n\nReading the dataset into RAM ...")

with open("../dataset/UNv1.0/after_step_3/en_train.json") as f:
    en_train = json.load(f)
        
with open("../dataset/UNv1.0/after_step_3/en_val.json") as f:
    en_val_input = json.load(f)
    
with open("../dataset/UNv1.0/after_step_3/fr_val.json") as f:
    fr_val_input = json.load(f)

print("Done")

print("\n\nCleaning validation set ...\n\n")
en_val_output=[]
fr_val_output=[]

   
for i in range(len(en_val_input)):
    
    utils.display_progress(i, len(en_val_input))
    
    if(en_val_input[i] not in en_train):
        en_val_output.append(en_val_input[i])
        fr_val_output.append(fr_val_input[i])


print("\n\nDone\n\n")


print("\n\nSaving the results ...")           
with open('../dataset/UNv1.0/en_val.json', 'w') as f:
    json.dump(en_val_output, f)
    
with open('../dataset/UNv1.0/fr_val.json', 'w') as f:
    json.dump(fr_val_output, f)
print("Done") 