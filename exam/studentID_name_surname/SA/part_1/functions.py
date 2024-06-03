# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn

# from conll import evaluate
from evals import evaluate_ote
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, model, device, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        assert sample['ids'].shape[0] == sample['mask'].shape[0] #== sample['y_slots'].shape[0] == sample['slots_len'].shape[0] == sample['intents'].shape[0]
        slots = model(sample['ids'], attention_mask=sample['mask'])
        slots = slots.permute(0,2,1) # to compute loss is necessary to permute
        loss_slot = criterion_slots(slots.to(device), sample['y_slots'].to(device))
        loss = loss_slot # In joint training we sum the losses. 
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array



def eval_loop(data, criterion_slots, model, lang, tokenizer, device):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots = model(sample['ids'], attention_mask=sample['mask'])
            # slots, intents = model(sample['utterances'], sample['slots_len'])
            slots = slots.permute(0,2,1)

            loss_slot = criterion_slots(slots.to(device), sample['y_slots'].to(device))
            # print("validation loss_slot = %.2f" % loss_slot)
            loss = loss_slot 
            loss_array.append(loss.item())
            
            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'][id_seq]
                utt_ids = sample['ids'][id_seq][1:length-1].tolist() # get the sequence without the padding 0s
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[1:length-1]]
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = seq[1:length-1].tolist()
                tmp_seq = []
                tmp_ref = []
                assert len(to_decode) == len(gt_slots)
                for id_el, slot_label in enumerate(gt_slots):
                    if slot_label == 'pad':
                        # If a word is split into multiple tokens, then it merges back into one single word-token
                        # for the sake of evaluation, e.g. tokenize(['whats'])=['[CLS]', 'what', "'", 's', '[SEP]']
                        # instead we want to put back together what's for the purpose of slot labeling
                        w = tmp_ref[-1][0]+utterance[id_el] # merge wordpieces. 
                        _, label = tmp_ref.pop() # pop incomplete word
                        tmp_ref.append((w, label)) # add newly constructed word
                        continue
                    tmp_ref.append((utterance[id_el], slot_label))
                    to_decode_id = to_decode[id_el]
                    tmp_seq.append((utterance[id_el], lang.id2slot[to_decode_id]))
                hyp_slots.append(tmp_seq)
                ref_slots.append(tmp_ref)

    try:            
         results = evaluate_ote(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
