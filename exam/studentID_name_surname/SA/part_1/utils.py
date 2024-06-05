# Add functions or classes used for data loading and preprocessing

import json
import torch
import torch.utils.data as data
import argparse
import random
from utils_e2e_tbsa import *

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
    def __init__(self, slots, pad_id, punct_id):
        self.pad_id = pad_id
        self.punct_id = punct_id
        self.slot2id = self.lab2id(slots)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
    
    def lab2id(self, elements):
        vocab = {}
        vocab['PAD'] = self.pad_id
        vocab['PUNCT'] = self.punct_id
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

def merge_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

class BertABSADataset(data.Dataset):
    """
    Custom PyTorch dataset class for intent classification and slot filling tasks.
    """

    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, tagging_scheme, pad_id, punct_id):
        """
        Initialize the dataset by mapping utterances, slots, and intents to integer IDs.

        :param dataset: List of dictionaries, each containing 'utterance', 'slots', and 'intent'.
        :param lang: Language object containing mapping information.
        :param unk: Unknown token (default is 'unk').
        """
        self.punct_id = punct_id
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.utterances = []
        self.utt_words = []
        self.slots = []
        self.unk = 'unk'
        
        # Map utterances, slots, and intents to integer IDs
        for x in dataset:
            self.utterances.append(x['sentence'])
            self.utt_words.append(x['words'])
            self.slots.append(x[tagging_scheme])
        
        self.slot_ids, self.utt_ids, self.utt_masks = self.mapping_seq(self.slots, self.utt_words, lang.slot2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """
        Return the dictionary for the example at index idx.

        :param idx: Index of the example to retrieve.
        :return: Dictionary containing 'utterance', 'slots', and 'intent' as integer IDs.
        """

        utt = self.utterances[idx]
        utt_words = self.utt_words[idx]
        slots = self.slot_ids[idx]
        ids = self.utt_ids[idx]
        mask = self.utt_masks[idx]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'slots' : torch.tensor(slots, dtype=torch.long), #.squeeze(), # unsqueeze so that we add an extra dim so that we can take .shape[1] in merge function
            'utt': utt,
            'utt_words': utt_words
        }
    
    def mapping_seq(self, data, utt_words, mapper): # Map sequences to number
        """
        Map a list of sequences to integer IDs, tokenizing each sequence.

        :param data: List of sequences.
        :param mapper: Mapper dictionary.
        :return: List of tokenized and mapped sequences.
        """

        res = []
        res_tokens = []
        res_masks = []
        # we have to tokenize the sentence following the target/word separation stored in utt_words
        for slots_list, words_list in zip(data, utt_words):
            tmp_seq = []
            sentence_tokens = []
            sentence_masks = []
            assert len(slots_list) == len(words_list), f"seq: {slots_list}, utt: {words_list}" # sanity check
            tokenized_words_list = self.tokenizer(words_list)
            
            tokens_list = tokenized_words_list['input_ids']
            masks_list = tokenized_words_list['attention_mask']

            for word_tokens, word_masks, label in zip(tokens_list, masks_list, slots_list):
                middle_tokens = word_tokens[1:-1]
                num_tokens = len(middle_tokens)
                if label in mapper:
                    tmp_seq.extend([mapper[label]] + [self.pad_id]*(num_tokens-1)) # only map the 1st token and add padding for the rest
                else:
                    raise ValueError(f"label: {label} not in mapper")
                    # tmp_seq.extend(mapper[self.unk])

                # use this for loop otherwise you have to write another for loop in __getitem__
                # remove the [CLS] and [SEP] tokens and only add them at the end of the sequence
                sentence_tokens.extend(middle_tokens)
                middle_masks = word_masks[1:-1]
                sentence_masks.extend(middle_masks)
            assert len(tmp_seq) == len(sentence_tokens) == len(sentence_masks)

            res.append([self.pad_id]+tmp_seq+[self.pad_id]) # add the [CLS] and [SEP] to the entire sequence
            res_tokens.append([self.pad_id]+sentence_tokens+[self.pad_id])
            res_masks.append([self.pad_id]+sentence_masks+[self.pad_id])
            # res_tokens.append([self.pad_id]+tokens_list+[self.pad_id])
            # res_masks.append([self.pad_id]+utt_masks_ids_sentence+[self.pad_id])
        return res, res_tokens, res_masks


def collate_fn(data, pad_id, device):

    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        # print(sequences)
        lengths = [max(seq.shape) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_ID (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_id)
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
    src_utt = src_utt.to(device)
    mask = mask.to(device)
    y_slots = y_slots.to(device)

    assert src_utt.shape == y_slots.shape == mask.shape, \
        f"src_utt: {src_utt.shape}, y_slots: {y_slots.shape}, mask: {mask.shape}"
    
    new_item["ids"] = src_utt
    new_item["mask"] = mask
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


def data_loader(parser):
    parser.add_argument("-ds_name", type=str, default='laptop14', help="dataset name")
    # dimension of LSTM hidden representations
    parser.add_argument("-dim_char", type=int, default=30, help="dimension of char embeddings")
    parser.add_argument("-dim_char_h", type=int, default=50, help="dimension of char hidden representations")
    parser.add_argument("-dim_ote_h", type=int, default=50, help="hidden dimension for opinion target extraction")
    parser.add_argument("-dim_ts_h", type=int, default=50, help="hidden dimension for targeted sentiment")
    parser.add_argument("-input_win", type=int, default=3, help="window size of input")
    parser.add_argument("-stm_win", type=int, default=3, help="window size of OE component")
    parser.add_argument("-optimizer", type=str, default="sgd", help="optimizer (or, trainer)")
    parser.add_argument("-n_epoch", type=int, default=40, help="number of training epoch")
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout rate for final representations")
    parser.add_argument("-emb_name", type=str, default="laptop14", help="name of word embedding")
    # Note: tagging schema is OT in the original data record
    parser.add_argument("-tagging_schema", type=str, default="BIEOS", help="tagging schema")
    parser.add_argument("-rnn_type", type=str, default="LSTM",
                        help="type of rnn unit, currently only LSTM and GRU are supported")
    parser.add_argument("-sgd_lr", type=float, default=0.1,
                        help="learning rate for sgd, only used when the optimizer is sgd")
    parser.add_argument("-clip_grad", type=float, default=5.0, help="maximum gradients")
    parser.add_argument("-lr_decay", type=float, default=0.05, help="decay rate of learning rate")
    parser.add_argument("-use_char", type=int, default=0, help="if use character-level word embeddings")
    parser.add_argument('-epsilon', type=float, default=0.5, help="maximum proportions of the boundary-based scores")
    
    
    dy_seed = 1314159
    random_seed = 1234
    #random_seed = 1972
    args = parser.parse_args()
    if args.ds_name == 'laptop14':
        random_seed = 13456
    if args.ds_name.startswith("twitter"):
        random_seed = 7788
    args.dynet_seed = dy_seed
    args.random_seed = random_seed
    random.seed(random_seed)
    emb_name = args.emb_name


    input_win = args.input_win
    stm_win = args.stm_win
    ds_name = args.ds_name
    tagging_schema = args.tagging_schema

    # build dataset
    train, train_map, val, val_map, test, test_map, vocab, char_vocab, ote_tag_vocab, ts_tag_vocab = build_dataset(
        ds_name=ds_name, input_win=input_win,
        tagging_schema=tagging_schema, stm_win=stm_win
    )

    return train, val, test, vocab, char_vocab, ote_tag_vocab, ts_tag_vocab