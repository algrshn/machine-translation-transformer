import json
import sys

# with open("dataset/UNv1.0/after_step_0/en_train.json") as f:
#     train= json.load(f)
    
# with open("dataset/UNv1.0/after_step_0/en_val.json") as f:
#     val= json.load(f)
    
    
with open("dataset/UNv1.0/after_step_3/en_train.json") as f:
    train= json.load(f)
    
with open("dataset/UNv1.0/en_val_reduced.json") as f:
    val= json.load(f)


    
failed=0
passed=0   
#for i in range(len(val)):
for i in range(1000):
    val_sent=val[i]
    try:
        k=train.index(val_sent)
        failed+=1
        print("FAILED")
    except:
        print(i)
        passed+=1
print("\n\n\n")        
print(failed)
print(passed)