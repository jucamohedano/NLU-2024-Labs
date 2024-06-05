# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from tqdm import tqdm
from functions import *
from model import ModelIAS
from functools import partial
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def init_args():
    parser = argparse.ArgumentParser(description="Bert training for Aspect Term Extraction (ATE)")
    parser.add_argument("--mode", type=str, default='eval', help="Model mode: train or eval")
    parser.add_argument("--use_wandb", type=str, default='true', help="Use wandb to log training results.")
    return parser

def init_wandb():
    import wandb
    # Set your Wandb token
    wandb_token = os.environ["WANDB_TOKEN"]

    # # Login to wandb
    wandb.login(key=wandb_token)

    # # Initialize wandb
    wandb.init(project="nlu-assignmet2-part1", allow_val_change=True)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'The device selected: {device}')

    # load data
    parser = init_args()
    model_path = 'bin/best_model.pt'

    mode = parser.parse_args().mode
    use_wandb = parser.parse_args().use_wandb
    if mode == 'train' and use_wandb == 'true':
        import wandb
        init_wandb()

    print(f'Running script in mode: {mode}. If you desire to change it use the --mode argument, i.e. python main.py --mode eval')

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
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=PAD_TOKEN, device=device))

    # Initialize model
    hid_size = 200
    emb_size = 300
    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    # hyperparams
    n_epochs = 200
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    PATIENCE = 3
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    config = None
    if mode == 'train' and use_wandb == 'true':
        #wandb: Define your config
        config = wandb.config
        config.epochs = n_epochs
        config.learning_rate = lr
        config.batch_size = BATCH_SIZE
        config.emb_size = emb_size
        config.hidden_size = hid_size
        config.patience = PATIENCE

    if mode == 'train':
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model, clip=clip)
            if use_wandb == 'true':
                # Log training loss to wandb
                wandb.log({"train_loss": np.asarray(loss).mean()})
            if x % 5 == 0: # We check the performance every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']
                if use_wandb == 'true':
                    # Log validation loss to wandb
                    wandb.log({"val_loss": np.asarray(loss_dev).mean()})
                    wandb.log({"f1": f1})
                # For decreasing the patience you can also use the average between slot f1 and intent accuracy
                if f1 > best_f1:
                    best_f1 = f1
                    model_info = {'state_dict': model.state_dict(), 'lang':lang}
                    torch.save(model_info, model_path)
                    patience = PATIENCE
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean
    else:
        print("*You are in evaluation mode*")
        # Load model
        checkpoint = torch.load(model_path)
        lang = checkpoint['lang']
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)    
        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])