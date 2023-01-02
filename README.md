# PyTorch implementation of Transformer from scratch from the original paper "Attention Is All You Need".

Originally published on [*my site*](https://alexgrishin.ai/pytorch_implementaion_of_attention_is_all_you_need).

<p>This is a PyTorch implementation of Transformer from the original paper <a href="https://arxiv.org/pdf/1706.03762.pdf" target="_blank">
"Attention Is All You Need"</a>.
Transformer is coded from scratch in "vanilla" PyTorch without use of PyTorch transformer classes.
The model was trained on UN English-French parallel corpus (about 8 mln sentence pairs after the original corpus of 11.4 mln pairs
was cleaned of too long and too short sentences) and can be used for translation of formal,
official documents (similar to what can be found in UN corpus) from English to French. The model stands no chance to understand
 a sentence like:<br><br>"Hey dude, we watched an awesome movie yesterday and then went bar hopping.".<br><br>
While it will do a good job translating a sentence like:<br><br>
"Despite very serious progress on battling deforestation achieved in the course of the last 12 years,
the area covered by forests is still shrinking, albeit at a slower rate."
<br><br> to French ("Malgré les progrès considérables accomplis au cours des 12 dernières années dans la lutte contre le déboisement,
la superficie des forêts continue de diminuer, mais à un rythme plus lent.").</p>

