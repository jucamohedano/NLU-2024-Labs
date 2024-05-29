# Add functions or classes used for data loading and preprocessing

import json
import torch
import torch.utils.data as data

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class Lang():
    def __init__(self, intents, slots, pad_token):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab


class BertIntentsAndSlots(data.Dataset):
    """
    Custom PyTorch dataset class for intent classification and slot filling tasks.
    """

    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, pad_token=0, unk='unk'):
        """
        Initialize the dataset by mapping utterances, slots, and intents to integer IDs.

        :param dataset: List of dictionaries, each containing 'utterance', 'slots', and 'intent'.
        :param lang: Language object containing mapping information.
        :param unk: Unknown token (default is 'unk').
        """
        self.unk = unk
        self.pad_token = pad_token
        self.tokenizer = tokenizer
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

        utt_inputs = self.tokenizer(utt, 
                               return_tensors="pt", 
                               return_token_type_ids=False)
                
        ids = utt_inputs['input_ids']
        mask = utt_inputs['attention_mask']

        return {
            'ids': ids.clone().long(), #torch.tensor(ids, dtype=torch.long),
            'mask': mask.clone().long(), #torch.tensor(mask, dtype=torch.long),
            'slots' : torch.tensor(slots, dtype=torch.long), #.squeeze(), # unsqueeze so that we add an extra dim so that we can take .shape[1] in merge function
            'intent': intent,
            'utt': utt
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
        for seq, utt in zip(data, self.utterances):
            tmp_seq = []
            # example of seq: 'O O O O O O O O B-fromloc.city_name O B-toloc.city_name'
            # example of utt: 'what is the cost for these flights from baltimore to philadelphia'
            assert len(seq.split()) == len(utt.split()), f"seq: {seq}, utt: {utt}" # sanity check
            for x, w in zip(seq.split(), utt.split()):
                tokenized_word = self.tokenizer(w)['input_ids'][1:-1] # don't take the [CLS] and [SEP] tokens
                if x in mapper:
                    tmp_seq.extend([mapper[x]] + [self.pad_token]*(len(tokenized_word)-1)) # only map the 1st token and add padding for the rest
                else:
                    tmp_seq.extend(mapper[self.unk])
                
            res.append([self.pad_token]+tmp_seq+[self.pad_token]) # add the [CLS] and [SEP] to the entire sequence
        return res


def collate_fn(data, pad_token, device):

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
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
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
    mask, _ = merge(new_item['mask'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"]).to(device)
    src_utt = src_utt.to(device)
    mask = mask.to(device)
    y_slots = y_slots.to(device)

    assert src_utt.shape == y_slots.shape == mask.shape, \
        f"src_utt: {src_utt.shape}, y_slots: {y_slots.shape}, mask: {mask.shape}"
    assert src_utt.shape[0] == intent.shape[0], f"intent: {intent.shape}"
    
    new_item["ids"] = src_utt
    new_item["mask"] = mask
    new_item["intent"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item