# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import ModelIAS
from utils import Lang

from model import *
from utils import *
from tqdm import tqdm

import copy
import numpy as np
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
import wandb
import os
from sklearn.model_selection import train_test_split
# Import everything from functions.py file
from functions import *

# Set your Wandb token
wandb_token = os.environ["WANDB_TOKEN"]

# Login to wandb
wandb.login(key=wandb_token)

# Initialize wandb
wandb.init(project="nlu-assignmet2-part1", allow_val_change=True)


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hid_size = 200
    emb_size = 300
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    PAD_TOKEN = 0

    # load data
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))


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

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, pad_token=0, cutoff=0)


    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    BATCH_SIZE =128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])