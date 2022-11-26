import torch
from torch import nn
from math import sqrt
import sys



class Model(nn.Module):

    def __init__(self, vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, full_positional_encoding_matrix, masking_minus_inf, device):
        super().__init__()
        
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.device=device
        self.full_positional_encoding_matrix=full_positional_encoding_matrix
        self.V=nn.Parameter(init_Xavier_tensor(input_dim=d_model, output_dim=vocab_size)) # Pre-softmax linear transformation after the decoder
        
        self.encoder=Encoder(N, d_model, d_ff, h, d_k, d_v, P_drop, device)
        self.decoder=Decoder(N, d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device)
        
        self.dropout_enc = nn.Dropout(p=P_drop)
        self.dropout_dec = nn.Dropout(p=P_drop)
        

    def forward(self, EN_onehot, FR_onehot_shifted):
        
        U=sqrt(self.d_model)*torch.transpose(self.V, 0, 1)        
        EN_embedding=torch.matmul(EN_onehot,U)
        
        encoder_input=self.dropout_enc(positional_encoding(EN_embedding, self.full_positional_encoding_matrix))
        
        encoder_output=self.encoder(encoder_input)
              
        FR_embedding=torch.matmul(FR_onehot_shifted,U)
        
        decoder_input=self.dropout_dec(positional_encoding(FR_embedding, self.full_positional_encoding_matrix))
        
        decoder_output = self.decoder(encoder_output, decoder_input)
        
        # FR_onehot_pred shape =(batch_size, vocab_size, fr_sent_len)
        FR_onehot_pred=torch.moveaxis(torch.matmul(decoder_output, self.V), 1, -1)

    
        return FR_onehot_pred
    


class Encoder(nn.Module):
    
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, P_drop, device):
        super().__init__()
               
        self.N=N
        
        for i in range(self.N):
            setattr(self, 'block_' + str(i), Enc_block(d_model, d_ff, h, d_k, d_v, P_drop, device))
        
    def forward(self, encoder_input):
                
        block_output_list=[]
        block_output_list.append(self.block_0(encoder_input))
        for i in range(1,self.N):
            block_output_list.append(getattr(self,'block_' + str(i))(block_output_list[i-1]))
                
        return block_output_list[-1]


class Decoder(nn.Module):
    
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device):
        super().__init__()
                
        self.N=N
        
        for i in range(self.N):
            setattr(self, 'block_' + str(i), Dec_block(d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device))
                
    def forward(self, encoder_output, decoder_input):
        
        block_output_list=[]
        block_output_list.append(self.block_0(encoder_output, decoder_input))
        for i in range(1,self.N):
            block_output_list.append(getattr(self,'block_' + str(i))(encoder_output, block_output_list[i-1]))
                
        return block_output_list[-1]
    
    
class Enc_block(nn.Module):
    
    def __init__(self, d_model, d_ff, h, d_k, d_v, P_drop, device):
        super().__init__()
        
        self.d_model=d_model
        self.mha=Multi_Head_Attention(d_model, h, d_k, d_v, device)
        self.ff=FF(d_model, d_ff, device)
        
        self.layer_norm1 = nn.LayerNorm(self.d_model, device=device)        
        self.layer_norm2 = nn.LayerNorm(self.d_model, device=device)
        self.dropout1 = nn.Dropout(p=P_drop)
        self.dropout2 = nn.Dropout(p=P_drop)
        
    def forward(self, block_input):
                
        mha_output=self.layer_norm1(block_input + self.dropout1(self.mha(block_input, block_input, block_input)))
        block_output=self.layer_norm2(mha_output + self.dropout2(self.ff(mha_output)))
            
        return block_output


class Dec_block(nn.Module):
    
    def __init__(self, d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device):
        super().__init__()
        
        self.d_model=d_model
        self.mmha=Multi_Head_Attention(d_model, h, d_k, d_v, device, masking_minus_inf=masking_minus_inf, masked=True)
        self.mha=Multi_Head_Attention(d_model, h, d_k, d_v, device)
        self.ff=FF(d_model, d_ff, device)
        
        self.layer_norm1 = nn.LayerNorm(self.d_model, device=device)
        self.layer_norm2 = nn.LayerNorm(self.d_model, device=device)
        self.layer_norm3 = nn.LayerNorm(self.d_model, device=device)
        self.dropout1 = nn.Dropout(p=P_drop)
        self.dropout2 = nn.Dropout(p=P_drop)
        self.dropout3 = nn.Dropout(p=P_drop)
        
    def forward(self, encoder_output, block_input):
        
        
        mmha_output=self.layer_norm1(block_input + self.dropout1(self.mmha(block_input, block_input, block_input)))
        mha_output=self.layer_norm2(mmha_output + self.dropout2(self.mha(mmha_output, encoder_output, encoder_output)))
        block_output=self.layer_norm3(mha_output + self.dropout3(self.ff(mha_output)))
            
        return block_output

 
class Multi_Head_Attention(nn.Module):
    
    def __init__(self, d_model, h, d_k, d_v, device, masking_minus_inf=None, masked=False):
        super().__init__()
        
        self.h=h
        self.masked=masked
        self.device=device
        self.masking_minus_inf=masking_minus_inf
        
        for i in range(self.h):
            setattr(self, 'Linear_Q_h' + str(i), nn.Linear(in_features=d_model, out_features=d_k, bias=False, device=device))
            setattr(self, 'Linear_K_h' + str(i), nn.Linear(in_features=d_model, out_features=d_k, bias=False, device=device))
            setattr(self, 'Linear_V_h' + str(i), nn.Linear(in_features=d_model, out_features=d_v, bias=False, device=device))
            
        self.Linear_after_concat=nn.Linear(in_features=d_v*h, out_features=d_model, bias=False, device=device)
        
    def forward(self, Q, K, V):
        
        Q_list=[]
        K_list=[]
        V_list=[]
        
        for i in range(self.h):
            Q_list.append(getattr(self,'Linear_Q_h' + str(i))(Q))
            K_list.append(getattr(self,'Linear_K_h' + str(i))(K))
            V_list.append(getattr(self,'Linear_V_h' + str(i))(V))
            
        SDPA_outputs_list=[]
        for i in range(self.h):
            SDPA_outputs_list.append(scaled_dot_product_attention(Q_list[i], K_list[i], V_list[i], self.device, self.masking_minus_inf, masked=self.masked))
            
        concat_output=torch.cat(SDPA_outputs_list, dim=-1)
        
        output=self.Linear_after_concat(concat_output)
            
        return output


class FF(nn.Module):
    
    def __init__(self, d_model, d_ff, device):
        super().__init__()
        
        self.d_model=d_model
        self.d_ff=d_ff
        
        layers=[]
        layers.append(nn.Linear(in_features=d_model, out_features=d_ff, device=device))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=d_ff, out_features=d_model, device=device))
                
        self.main = nn.Sequential(*layers)
        
    def forward(self, ff_input):
        
        output=self.main(ff_input)
            
        return output
   
def scaled_dot_product_attention(Q, K, V, device, masking_minus_inf, masked):
    
    QK=torch.matmul(Q,torch.transpose(K, -2, -1))/sqrt(K.shape[-1])
    
    if(masked):
        mask_to_add=masking_minus_inf*torch.triu(torch.ones((1, QK.shape[1], QK.shape[2])).to(device), diagonal=1)
        QK_masked=QK+mask_to_add
        # QK_masked=QK        
                
    else:
        QK_masked=QK
    
    output=torch.matmul(nn.functional.softmax(QK_masked, dim=-1),V)
        
    return output


def positional_encoding(embedding, full_positional_encoding_matrix):
    
    PE=full_positional_encoding_matrix[None,:embedding.shape[1],:]
    
    try:
        input_for_encoder_or_decoder = embedding + PE
    except:
        print(f"Embedding: {embedding.shape}")
        print(f"PE: {PE.shape}")
        sys.exit()
        
    return input_for_encoder_or_decoder

        
    
def init_Xavier_tensor(input_dim, output_dim):  # for tanh
    
    W=torch.empty((input_dim, output_dim), dtype=torch.float32).normal_(std=sqrt(2/(input_dim+output_dim)))
    
    return W