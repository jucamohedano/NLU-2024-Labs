# Global variables
import os
device = "cuda:0" #means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0
PRINT_DATA_STATS=0

lr = 0.0001 # learning rate
clip = 5 # Clip the gradient

import json
from pprint import pprint

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

if PRINT_DATA_STATS:
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))


import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# First we get the 10% of the training set, then we compute the percentage of these examples 

portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
count_y = Counter(intents)

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # If some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])
# Random Stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]


# Intent distributions
if PRINT_DATA_STATS:
    print('Train:')
    pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    print('Dev:'), 
    pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    print('Test:') 
    pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    print('='*89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))


from collections import Counter

w2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
slot2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
intent2id = {}

from collections import Counter
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab


words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                        # however this depends on the research purpose
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(words, intents, slots, cutoff=0)
out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# BERT model script from: huggingface.co
from transformers import BertTokenizer, BertModel
from pprint import pprint



# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch
import torch.utils.data as data

class BertIntentsAndSlots(data.Dataset):
    """
    Custom PyTorch dataset class for intent classification and slot filling tasks.
    """

    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        """
        Initialize the dataset by mapping utterances, slots, and intents to integer IDs.

        :param dataset: List of dictionaries, each containing 'utterance', 'slots', and 'intent'.
        :param lang: Language object containing mapping information.
        :param unk: Unknown token (default is 'unk').
        """

        # Setting based on the current model type
        # cls_token = tokenizer.cls_token
        # sep_token = tokenizer.sep_token
        # unk_token = tokenizer.unk_token
        # pad_token_id = tokenizer.pad_token_id

        self.utterances = []
        self.intents = []
        self.slots = []
        
        # Map utterances, slots, and intents to integer IDs
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
        
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """
        Return the dictionary for the example at index idx.

        :param idx: Index of the example to retrieve.
        :return: Dictionary containing 'utterance', 'slots', and 'intent' as integer IDs.
        """
        utt = self.utterances[idx]
        slots = self.slot_ids[idx]

        intent = self.intent_ids[idx]

        utt_inputs = tokenizer(utt, 
                               return_tensors="pt", 
                            #    padding=True,
                            #    max_length=200,
                            #    pad_to_max_length=True,
                               return_token_type_ids=True)
                
        ids = utt_inputs['input_ids']
        mask = utt_inputs['attention_mask']
        token_type_ids = utt_inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'slots' : torch.tensor(slots, dtype=torch.long), #.squeeze(), # unsqueeze so that we add an extra dim so that we can take .shape[1] in merge function
            'intent': intent
        }
    
    def mapping_lab(self, data, mapper):
        """
        Map a list of labels to integer IDs, using the unknown token ID if not found in mapper.

        :param data: List of labels.
        :param mapper: Mapper dictionary.
        :return: List of integer IDs.
        """
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        """
        Map a list of sequences to integer IDs, tokenizing each sequence.

        :param data: List of sequences.
        :param mapper: Mapper dictionary.
        :return: List of tokenized and mapped sequences.
        """
        res = []
        # print(data)
        # print(100*"y")
        # print(self.utterances)
        for seq, utt in zip(data, self.utterances):
            tmp_seq = []
            # example of slot: print(seq)= O O O O O O O O B-fromloc.city_name O B-toloc.city_name
            for x, w in zip(seq.split(), utt.split()):
                tokenized_word = tokenizer(w)['input_ids'][1:-1] # don't take the [CLS] and [SEP] tokens
                # if len(tokenized_word) > 1:
                #     print(tokenized_word, tokenizer.convert_ids_to_tokens(tokenized_word))
                if x in mapper:
                    tmp_seq.extend([mapper[x]] + [PAD_TOKEN]*(len(tokenized_word)-1)) # Example: [ [ID][PAD][PAD][PAD] ]
                    # if len(tokenized_word) > 1:
                    #     print([mapper[x]] + [PAD_TOKEN]*(len(tokenized_word)-1))
                else:
                    tmp_seq.extend(mapper[self.unk])
                
            res.append([PAD_TOKEN]+tmp_seq+[PAD_TOKEN]) # add the [CLS] and [SEP] to the entire sequence
        return res


train_dataset = BertIntentsAndSlots(train_raw, lang)
dev_dataset = BertIntentsAndSlots(dev_raw, lang)
test_dataset = BertIntentsAndSlots(test_raw, lang)

def collate_fn(data):

    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        # print(sequences)
        lengths = [max(seq.shape) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        # print(type(padded_seqs), type(lengths), len(padded_seqs), len(lengths))
        return padded_seqs, lengths
    
    # create an empty array of -100 of length max_length
    # encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['ids']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['ids'])
    src_utt.to(device)
    mask, _ = merge(new_item['mask'])
    mask.to(device)
    token_type_ids, _ = merge(new_item['token_type_ids'])
    token_type_ids.to(device)
    y_slots, y_lengths = merge(new_item["slots"])
    y_slots.to(device)
    # y_lengths.to(device)
    intent = torch.LongTensor(new_item["intent"]).to(device)
    intent.to(device)
    # print(100*'*')
    # print(y_slots.shape)
    # print(100*'*')
    
    new_item["ids"] = src_utt
    new_item["mask"] = mask
    new_item["token_type_ids"] = token_type_ids
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)



# Freeze BERT layers and replace top layers
# for param in bert_model.parameters():
#     param.requires_grad = False

class BertJoint(nn.Module):
    def __init__(self):
        super(BertJoint, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Define output layers for multi-task learning
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, out_slot) # token-label classifcation head
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, out_int) # sentence classification head
    
    def forward(self, input_ids, attention_mask, token_type_ids,):
        slots, intent = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False) # sequence_output, pooled_output, (hidden_states), (attentions)

        slot_logits = self.slot_classifier(slots)
        intent_logits = self.intent_classifier(intent)
        return slot_logits, intent_logits


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['ids'], attention_mask=sample['mask'], token_type_ids=sample['token_type_ids'])
        # _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        # print(slots.shape, intent.shape)
        # print(sample['y_slots'].shape, sample['intents'].shape)
        slots = slots.permute(0,2,1) # to compute loss is necessary to permute
        loss_intent = criterion_intents(intent.to(device), sample['intents'])
        loss_slot = criterion_slots(slots.to(device), sample['y_slots'].to(device))
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array


from conll import evaluate
from sklearn.metrics import classification_report
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['ids'], attention_mask=sample['mask'], token_type_ids=sample['token_type_ids'])
            # slots, intents = model(sample['utterances'], sample['slots_len'])
            slots = slots.permute(0,2,1)

            loss_intent = criterion_intents(intents.to(device), sample['intents'])
            loss_slot = criterion_slots(slots.to(device), sample['y_slots'].to(device))
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            # print(intents.shape)
            # print(torch.argmax(intents, dim=1))

            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            print(output_slots.shape)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'][id_seq]
                utt_ids = sample['ids'][id_seq][:length].tolist() # get the sequence without the padding 0s
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                # utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
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


# Define optimizer
import torch.optim as optim
from tqdm import tqdm

bert_model = BertJoint()
optimizer = optim.Adam(bert_model.parameters(), lr=0.0001)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()

n_epochs = 25
patience = 5
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0


for x in tqdm(range(1,n_epochs)):
    loss = train_loop(train_loader, optimizer, criterion_slots, 
                      criterion_intents, bert_model, clip=clip)
    if x % 5 == 0: # We check the performance every 5 epochs
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, bert_model, lang)
        losses_dev.append(np.asarray(loss_dev).mean())
        
        f1 = results_dev['total']['f']
        # For decreasing the patience you can also use the average between slot f1 and intent accuracy
        if f1 > best_f1:
            best_f1 = f1
            # Here you should save the model
            patience = 5
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                         criterion_intents, bert_model, lang)    
print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])