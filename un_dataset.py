from torch.utils.data import Dataset
from tokenizers import Tokenizer
import numpy as np
import pickle

with open('dataset/UNv1.0/en_train.pickle', 'rb') as f:
    en_train = pickle.load(f)
    
with open('dataset/UNv1.0/fr_train.pickle', 'rb') as f:
    fr_train = pickle.load(f)

tokenizer = Tokenizer.from_file("dataset/UNv1.0/tokenizer.json")
bos_token_id=tokenizer.token_to_id("[BOS]")

vocab_size=tokenizer.get_vocab_size()


class UN_Dataset(Dataset):
    def __init__(self):
        self.num_of_batches = len(en_train)

    def __len__(self):
                                    
        return self.num_of_batches

    def __getitem__(self, idx):
        
        EN=en_train[idx]
        FR=fr_train[idx]
        
        assert EN.shape[0] == FR.shape[0]
        
        FR_shifted=np.zeros_like(FR)
        FR_shifted[:,1:]=FR[:,:-1]
        FR_shifted[:,0]=bos_token_id
                
        batch_size=EN.shape[0]
        
        en_batch_sentence_len=EN.shape[1]
        fr_batch_sentence_len=FR.shape[1]
        
        EN_onehot=np.zeros((batch_size, en_batch_sentence_len, vocab_size), dtype=np.float32)
        FR_onehot_shifted=np.zeros((batch_size, fr_batch_sentence_len, vocab_size), dtype=np.float32)

        for batch_num in range(batch_size):            
            for pos in range(en_batch_sentence_len):
                EN_onehot[batch_num, pos, EN[batch_num,pos]]=1
            for pos in range(fr_batch_sentence_len):
                FR_onehot_shifted[batch_num, pos, FR_shifted[batch_num,pos]]=1
                
        FR_int=FR.astype(np.int)
            
        return EN_onehot, FR_onehot_shifted, FR_int