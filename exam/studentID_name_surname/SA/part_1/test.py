# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import Lang
from utils import *
from model import *
from functions import *
from tqdm import tqdm
from functools import partial
from collections import Counter
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# load data
train, val, test, vocab, _, ote_tag_vocab, ts_tag_vocab = data_loader()

# PAD_TOKEN = vocab['PADDING']
# PUNCT_TOKEN = vocab['PUNCT']
PUNCT_TOKEN = PAD_TOKEN = 0
slots = list(ote_tag_vocab.keys())

lang = Lang(slots, pad_token=PAD_TOKEN, punct_token=PUNCT_TOKEN)
out_slot = len(lang.slot2id)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# num_added_toks = tokenizer.add_tokens(["PUNCT"])
# print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.


# ote_tags because we only want to predict the aspect, ts_tags also predict sentiment
train_dataset = BertABSA(train, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_token=PAD_TOKEN, punct_token=PUNCT_TOKEN)
dev_dataset = BertABSA(val, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_token=PAD_TOKEN, punct_token=PUNCT_TOKEN)
test_dataset = BertABSA(test, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_token=PAD_TOKEN, punct_token=PUNCT_TOKEN)

BATCH_SIZE = 8
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device),  shuffle=True)
# dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))
bert_model = BertJoint(out_slot).to(device)
# bert_model.resize_token_embeddings(len(tokenizer)) # due to adding a new token

# hyperparams
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
n_epochs = 100
CLIP = 5
PATIENCE = 3
lr = 0.00001

# Define optimizer
optimizer = optim.Adam(bert_model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)



# for x in tqdm(range(1,n_epochs)):
#     loss = train_loop(train_loader, optimizer, criterion_slots, 
#                     bert_model, device, clip=CLIP)
#     # Log training loss to wandb
#     # wandb.log({"train_loss": np.asarray(loss).mean()})
#     if x % 5 == 0: # We check the performance every 5 epochs
#         sampled_epochs.append(x)
#         losses_train.append(np.asarray(loss).mean())
#         results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
#                                                         bert_model, lang, tokenizer, device)
#         losses_dev.append(np.asarray(loss_dev).mean())
        
#         f1 = results_dev['total']['f']
#         print('Validation Slot F1: ', results_dev['total']['f'])
#         print('Validation Intent Accuracy:', intent_res['accuracy'])
        
        # Log validation loss to wandb
        # wandb.log({"val_loss": np.asarray(loss_dev).mean()})
        # wandb.log({"f1": f1})
        
        # For decreasing the PATIENCE you can also use the average between slot f1 and intent accuracy
        # if f1 > best_f1:
        #     best_f1 = f1
        #     # Here you should save the model
        #     patience = PATIENCE
        # else:
        #     patience -= 1
        # if patience <= 0: # Early stopping with patience
        #     print("no more patience, finishing training")
        #     break # Not nice but it keeps the code clean

results_test, _ = eval_loop(test_loader, criterion_slots, 
                                         bert_model, lang, tokenizer, device)    
# print('Slot F1: ', results_test['total']['f'])
# print('Intent Accuracy:', intent_test['accuracy'])
