import torch
import torch.nn.functional as F
import sys
import math
import numpy as np
from copy import deepcopy
   

def display_progress(batch_num, num_of_batches):
    
    total=num_of_batches-1
    bar_len = 60
    filled_len = int(round(bar_len * batch_num / float(total)))

    percents = round(100.0 * batch_num / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()    
    
    
# prepares a numpy array with shape = (positional_encoding_max_pos, d_model)
def prepare_full_positional_encoding_matrix(d_model, positional_encoding_wavelength_scale, positional_encoding_max_pos):
    
    PE=np.zeros((positional_encoding_max_pos, d_model))
    
    for pos in range(positional_encoding_max_pos):
        for dim in range(d_model):
            if(dim % 2 == 0):
                PE[pos,dim]=math.sin(pos/(positional_encoding_wavelength_scale**(dim/d_model)))
            elif(dim % 2 == 1):
                PE[pos,dim]=math.cos(pos/(positional_encoding_wavelength_scale**((dim-1)/d_model)))
    
    
    return PE

def inference_greedy_search(model, EN_onehot, device, vocab_size, bos_token_id, pad_token_id, max_len):
    
    fr_sentence_pred=[]
    keep_predicting_next_token=True
    while keep_predicting_next_token:
        
        FR_onehot_shifted_np=np.zeros((1,len(fr_sentence_pred)+1,vocab_size), dtype=np.float32)
        FR_onehot_shifted_np[0,0,bos_token_id]=1
        
        for pos in range(len(fr_sentence_pred)):
            FR_onehot_shifted_np[0,pos+1,int(fr_sentence_pred[pos])]=1
            
        FR_onehot_shifted=torch.tensor(FR_onehot_shifted_np).to(device)
        
        FR_onehot_pred = model(EN_onehot, FR_onehot_shifted)
    
        pred = torch.argmax(F.softmax(torch.squeeze(FR_onehot_pred, dim=0), dim=0), dim=0).detach().cpu().numpy()[-1]
        
        if(int(pred)==pad_token_id or len(fr_sentence_pred)==max_len):
            keep_predicting_next_token=False
        else:
            fr_sentence_pred.append(pred)
            
    return fr_sentence_pred

def inference_beam_search(model, EN_onehot, device, vocab_size, bos_token_id, pad_token_id, max_len, beam_size, length_penalty):
    
    final_candidates=[]
    final_candidates_log_prob=[]
    candidates=[[]]
    candidates_log_prob=[0]
    
    stop_predicting=False
    for i in range(max_len):
        
        if(stop_predicting):
            break
                
        candidates_buffer=[]
        candidates_log_prob_buffer=[]
        for j in range(len(candidates)):
            
            fr_sentence_pred=candidates[j]
            fr_sentence_log_prob=candidates_log_prob[j]
                   
            FR_onehot_shifted_np=np.zeros((1,len(fr_sentence_pred)+1,vocab_size), dtype=np.float32)
            FR_onehot_shifted_np[0,0,bos_token_id]=1
            
            for pos in range(len(fr_sentence_pred)):
                FR_onehot_shifted_np[0,pos+1,int(fr_sentence_pred[pos])]=1
                
            FR_onehot_shifted=torch.tensor(FR_onehot_shifted_np).to(device)
            
            FR_onehot_pred = model(EN_onehot, FR_onehot_shifted)
        
            val, ind = torch.topk(F.softmax(torch.squeeze(FR_onehot_pred, dim=0), dim=0), k=beam_size, dim=0)
            
            pred_list=ind[:,-1].detach().cpu().numpy().tolist()
            prob_list=val[:,-1].detach().cpu().numpy().tolist()
            
            
            for k in range(len(pred_list)):
                pred=pred_list[k]
                prob=prob_list[k]
                buffer=deepcopy(fr_sentence_pred)
                buffer.append(pred)
                candidates_buffer.append(buffer)
                candidates_log_prob_buffer.append(fr_sentence_log_prob+math.log(prob))
        
        zipped=sorted(zip(candidates_log_prob_buffer,candidates_buffer), reverse=True)        
        candidates_log_prob_buffer,candidates_buffer=zip(*zipped)
    
        candidates=[]
        candidates_log_prob=[]
        for k in range(beam_size):
            candidate=candidates_buffer[k]
            candidate_log_prob=candidates_log_prob_buffer[k]
            if(candidate[-1]==pad_token_id):
                final_candidates.append(candidate)
                final_candidates_log_prob.append(candidate_log_prob)
                # if(len(final_candidates)==beam_size):
                #     stop_predicting=True
                #     break
            else:
                candidates.append(candidate)
                candidates_log_prob.append(candidate_log_prob)
        if(len(candidates)==0):
            stop_predicting=True
            break
                
            
    final_candidates_score=[]        
    for k in range(len(final_candidates)):
        candidate_len=len(final_candidates[k])
        lp=((5+candidate_len)/6)**length_penalty
        candidate_score=final_candidates_log_prob[k]/lp
        final_candidates_score.append(candidate_score)
    
    zipped=sorted(zip(final_candidates_score,final_candidates), reverse=True)    
    final_candidates_score,final_candidates=zip(*zipped)
    
    
    return  final_candidates[0][:-1]       


