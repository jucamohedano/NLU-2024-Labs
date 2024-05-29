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


# Set your Wandb token
wandb_token = os.environ["WANDB_TOKEN"]

# # Login to wandb
wandb.login(key=wandb_token)

# # Initialize wandb
wandb.init(project="nlu-assignmet2-part2-bert", allow_val_change=True)

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    lang = Lang(intents, slots, pad_token=0)
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = BertIntentsAndSlots(train_raw, lang, tokenizer=tokenizer, pad_token=PAD_TOKEN)
    dev_dataset = BertIntentsAndSlots(dev_raw, lang, tokenizer=tokenizer, pad_token=PAD_TOKEN)
    test_dataset = BertIntentsAndSlots(test_raw, lang, tokenizer=tokenizer, pad_token=PAD_TOKEN)

    # Dataloader instantiations
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))

    # Create model
    bert_model = BertJoint(out_slot, out_int)

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
    criterion_intents = nn.CrossEntropyLoss()

    #wandb: Define your config
    config = wandb.config
    config.epochs = n_epochs
    config.learning_rate = lr
    config.batch_size = BATCH_SIZE
    config.patience = PATIENCE

    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, bert_model, device, clip=CLIP)
        # Log training loss to wandb
        wandb.log({"train_loss": np.asarray(loss).mean()})
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, bert_model, lang, tokenizer, device)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            print('Validation Slot F1: ', results_dev['total']['f'])
            print('Validation Intent Accuracy:', intent_res['accuracy'])
            
            # Log validation loss to wandb
            wandb.log({"val_loss": np.asarray(loss_dev).mean()})
            wandb.log({"f1": f1})
            
            # For decreasing the PATIENCE you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = PATIENCE
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                print("no more patience, finishing training")
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, bert_model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])
    # Log test metrics to wandb

    # To save the model
    path = 'bin/best_model.pt'
    torch.save(bert_model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))