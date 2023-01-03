# PyTorch implementation of Transformer from scratch from the original paper "Attention Is All You Need".

Originally published on [*my site*](https://alexgrishin.ai/pytorch_implementaion_of_attention_is_all_you_need).
<br /><br />

This is a PyTorch implementation of Transformer from the original paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
Transformer is coded from scratch in "vanilla" PyTorch without use of PyTorch transformer classes.
The model was trained on UN English-French parallel corpus (about 8 mln sentence pairs after the original corpus of 11.4 mln pairs
was cleaned of too long and too short sentences) and can be used for translation of formal,
official documents (similar to what can be found in UN corpus) from English to French. The model stands no chance to understand
 a sentence like:

"Hey dude, we watched an awesome movie yesterday and then went bar hopping.".

While it will do a good job translating a sentence like:

"Despite very serious progress on battling deforestation achieved in the course of the last 12 years,
the area covered by forests is still shrinking, albeit at a slower rate."

to French ("Malgré les progrès considérables accomplis au cours des 12 dernières années dans la lutte contre le déboisement,
la superficie des forêts continue de diminuer, mais à un rythme plus lent.").

The model achieves the BLEU score of 0.43 on the validation
set and exhibits close to human quality of translation on [15 long and convoluted test sentences](https://github.com/algrshn/machine-translation-transformer/blob/main/dataset/UNv1.0/other/my_own_15_sentences.txt)
(like the one above) I thought up myself.
With Google Translate English->French translation serving as reference for those 15 sentences,
the model achieves the BLEU score of 0.57. The model's translations can be viewed [here](https://github.com/algrshn/machine-translation-transformer/blob/main/dataset/UNv1.0/other/my_own_15_sentences_with_my_translation_greedy_search_epoch_14.txt).

My another takeaway is that I'm not certain the BLEU scores should always be trusted blindly to judge the quality of translation.
I hired a native French speaker to compare quality of translation the model produced after each epoch of training (the model was trained
for 42 epochs). Although the BLEU score on validation set showed no signs of going down, the expert chose epoch 14 as the best and
clearly indentified that
quality started to deteriorate noticeably after 20th epoch and was significantly lower at the end.<br /><br />

### Preprocessing data

The original UN English-French parallel corpus consists of roughly 11.4 mln sentence pairs. The corresponding files
(orig.en and orig.fr) can be downloaded
from my google drive:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;orig.en - [google drive link](https://drive.google.com/file/d/1Wf3osSE6FV659H5KgM9IBtSd5_39hpFd)<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;orig.fr - [google drive link](https://drive.google.com/file/d/1dkMh9xnxEBzcsgBi4jbYeFNmuhLnBOAB)

Download and place them in dataset/UNv1.0/ folder.

Configuration file config.txt that controls preprocessing is in preprocessing/ folder. This is how it looks:

<div>
\[convert_dataset\]<br>
reduction_len=40<br>
N_val=50000<br>
<br>
\[train_tokenizer\]<br>
vocab_size=37000<br>
<br>
\[remove_too_long_too_short\]<br>
max_len=99<br>
min_len=3<br>
<br>
\[split_into_batches\]<br>
approx_num_of_src_tokens_in_batch=1050<br>
approx_num_of_trg_tokens_in_batch=1540<br>
</div>
