def add_another_sentence_to_batch_or_close_batch(approx_num_of_src_tokens_in_batch, approx_num_of_trg_tokens_in_batch, en_curr_max_len_in_batch, fr_curr_max_len_in_batch, curr_num_of_sentences_in_batch, next_en_sentence, next_fr_sentence):
    
    before=(en_curr_max_len_in_batch*curr_num_of_sentences_in_batch, fr_curr_max_len_in_batch*curr_num_of_sentences_in_batch)
    
    en_new_max_len_in_batch=max(en_curr_max_len_in_batch, len(next_en_sentence))
    fr_new_max_len_in_batch=max(fr_curr_max_len_in_batch, len(next_fr_sentence))
    
    after=(en_new_max_len_in_batch*(curr_num_of_sentences_in_batch+1), fr_new_max_len_in_batch*(curr_num_of_sentences_in_batch+1))
    
    optimum=(approx_num_of_src_tokens_in_batch, approx_num_of_trg_tokens_in_batch)
    
    decision=add_or_not(before, after, optimum)


    if(decision=="yes"):
        
        output = "add_to_current_batch"
    
    elif(decision=="no"):
        
        output = "close_current_batch_and_add_to_next"
            
    return output

def add_or_not(before, after, optimum):
            
    if(after[0]>optimum[0] or after[1]>optimum[1]):
        
        output = "no"
        
    else:
        
        output = "yes"
        
    return output
    
    
def get_max_length_in_batch(batch):
    
    len_list=[len(sentence) for sentence in batch]
    
    max_len=max(len_list)
    
    return max_len

            
    
    
    
    
    
    
    
    