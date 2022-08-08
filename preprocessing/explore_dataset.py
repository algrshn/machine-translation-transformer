import random

ds_type="test" # can be either train, dev or test

with open('../dataset/UNv1.0/' + ds_type  +'.en', 'r') as f:
    en_list = f.readlines()
    
with open('../dataset/UNv1.0/' + ds_type  + '.fr', 'r') as f:
    fr_list = f.readlines()
    

N=len(en_list)

while 1:
    x=input("Press enter to proceed, q to quit: ")
    if(x=='q'):
        break
    
    n=random.randint(0, N-1)
    print(n)
    print(en_list[n])
    print(fr_list[n])
    print("\n-------------\n")
    
