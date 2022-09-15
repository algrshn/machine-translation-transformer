import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class Model(nn.Module):

    def __init__(self, vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop):
        super().__init__()
        
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.V=nn.Parameter(init_Xavier_tensor(input_dim=d_model, output_dim=vocab_size)) # Pre-softmax linear transformation after the decoder
        
        self.encoder=Encoder(N, d_model, d_ff, h, d_k, d_v, P_drop)
        self.decoder=Decoder(N, d_model, d_ff, h, d_k, d_v, P_drop)

    def forward(self, EN_onehot, FR_onehot_shifted):
        
        # EN_embedding shape =(en_sent_len, batch_size, d_model)
        U=sqrt(self.d_model)*torch.transpose(self.V, 0, 1)        
        EN_embedding=torch.matmul(torch.transpose(EN_onehot,0,1),U)
        
        encoder_input=positional_encoding(EN_embedding)
        
        encoder_output=self.encoder(encoder_input)
        
        # FR_embedding shape =(fr_sent_len, batch_size, d_model)        
        FR_embedding=torch.matmul(torch.transpose(FR_onehot_shifted,0,1),U)
        
        decoder_input=positional_encoding(FR_embedding)
        
        # decoder_output shape =(fr_sent_len, batch_size, d_model)
        decoder_output = self.decoder(encoder_output, decoder_input)
        
        # FR_onehot_pred shape =(batch_size, vocab_size, fr_sent_len)
        FR_onehot_pred=torch.moveaxis(torch.matmul(decoder_output, self.V), 0, -1)

    
        return FR_onehot_pred
    


class Encoder(nn.Module):
    
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, P_drop):
        super().__init__()
        
        self.N=N
        self.d_model=d_model
        self.d_ff=d_ff
        self.h=h
        self.d_k=d_k
        self.d_v=d_v
        self.P_drop=P_drop
        
    def forward(self, encoder_input):
        
        encoder_output=encoder_input
            
        return encoder_output


class Decoder(nn.Module):
    
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, P_drop):
        super().__init__()
        
        self.N=N
        self.d_model=d_model
        self.d_ff=d_ff
        self.h=h
        self.d_k=d_k
        self.d_v=d_v
        self.P_drop=P_drop
        
    def forward(self, encoder_output, decoder_input):
        
        decoder_output=decoder_input
            
        return decoder_output


def positional_encoding(embedding):
    
    input_for_encoder_or_decoder = embedding
    
    return input_for_encoder_or_decoder
    
    
def init_Xavier_tensor(input_dim, output_dim):  # for tanh
    
    W=torch.empty((input_dim, output_dim), dtype=torch.float32).normal_(std=sqrt(2/(input_dim+output_dim)))
    
    return W