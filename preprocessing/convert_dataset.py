import random
import json
import configparser

config=configparser.ConfigParser()
config.read('config.txt')

reduction_len=int(config.get('preprocessing', 'reduction_len'))
N_val=int(config.get('preprocessing', 'N_val'))


with open('../dataset/UNv1.0/orig.en', 'r') as f:
    en_list= f.readlines()
    
with open('../dataset/UNv1.0/orig.fr', 'r') as f:
    fr_list = f.readlines()


en_lengths=[]
fr_lengths=[]
        
for en in en_list:
    en_lengths.append(len(en.split(" ")))

for fr in fr_list:
    fr_lengths.append(len(fr.split(" ")))

en_reduced=[]
fr_reduced=[]

for i in range(len(en_list)):
    
    if(en_lengths[i]<reduction_len and fr_lengths[i]<reduction_len):
        en_reduced.append(en_list[i])
        fr_reduced.append(fr_list[i])
        
c = list(zip(en_reduced, fr_reduced))
random.shuffle(c)
en_reduced, fr_reduced = zip(*c)

en_train=[]
fr_train=[]

en_val=[]
fr_val=[]


for i in range(N_val):
    en_val.append(en_reduced[i])
    fr_val.append(fr_reduced[i])
    
for i in range(N_val, len(en_reduced)):
    en_train.append(en_reduced[i])
    fr_train.append(fr_reduced[i])    
    

with open('../dataset/UNv1.0/en_train.json', 'w') as f:
    json.dump(en_train, f)
    
with open('../dataset/UNv1.0/fr_train.json', 'w') as f:
    json.dump(fr_train, f)
    
with open('../dataset/UNv1.0/en_val.json', 'w') as f:
    json.dump(en_val, f)
    
with open('../dataset/UNv1.0/fr_val.json', 'w') as f:
    json.dump(fr_val, f)