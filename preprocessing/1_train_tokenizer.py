# takes train files en_train.json and fr_train.json,
# trains one shared tokenizer (with shared vocabulary),
# and saves it to tokenizer.json

import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import decoders
import configparser

config=configparser.ConfigParser()
config.read('config.txt')

vocab_size=int(config.get('train_tokenizer', 'vocab_size'))

with open('../dataset/UNv1.0/after_step_0/en_train.json') as f:
    en_train = json.load(f)
    
with open('../dataset/UNv1.0/after_step_0/fr_train.json') as f:
    fr_train = json.load(f)
    
combined_train=en_train + fr_train
    
    
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]"])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = decoders.ByteLevel()


tokenizer.train_from_iterator(combined_train, trainer)


tokenizer.save("../dataset/UNv1.0/tokenizer.json")
