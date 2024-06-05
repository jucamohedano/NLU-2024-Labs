# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os
import wandb
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import Lang
from utils import *
from model import *
from functions import *
from tqdm import tqdm
from functools import partial
from transformers import BertTokenizer
from torch.utils.data import DataLoader


def init_args():
    parser = argparse.ArgumentParser(description="Bert training for ABSA task")
    parser.add_argument("-mode", type=str, default='eval', help="Model mode: train or eval")
    return parser

def init_wandb():
    # Set your Wandb token
    wandb_token = os.environ["WANDB_TOKEN"]

    # # Login to wandb
    wandb.login(key=wandb_token)

    # # Initialize wandb
    wandb.init(project="nlu-assignmet3-part1", allow_val_change=True)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'The device selected: {device}')

    # load data
    parser = init_args()
    model_path = 'bin/best_model.pt'

    mode = parser.parse_args().mode
    if mode == 'train':
        init_wandb()
    print(f'Running script in mode: {mode}. If you desire to change it use the -mode argument, i.e. python main.py -mode eval')
    train, val, test, vocab, _, ote_tag_vocab, _ = data_loader(parser)

    PAD_ID = vocab['PADDING']
    PUNCT_ID = vocab['PUNCT']
    slots = list(ote_tag_vocab.keys())

    lang = Lang(slots, pad_id=PAD_ID, punct_id=PUNCT_ID)
    out_slot = len(lang.slot2id)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ote_tags because we only want to predict the aspect, ts_tags also predict sentiment
    train_dataset = BertABSADataset(train, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_id=PAD_ID, punct_id=PUNCT_ID)
    dev_dataset = BertABSADataset(val, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_id=PAD_ID, punct_id=PUNCT_ID)
    test_dataset = BertABSADataset(test, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_id=PAD_ID, punct_id=PUNCT_ID)

    # Dataloader instantiations
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_id=PAD_ID, device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_id=PAD_ID, device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_id=PAD_ID, device=device))
    
    # Create model
    bert_model = BertABSAModel(out_slot).to(device)
    
    # hyperparams
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    n_epochs = 100
    CLIP = 5
    PATIENCE = 3
    patience = PATIENCE
    lr = 0.00001

    # Define optimizer
    optimizer = optim.Adam(bert_model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    config = None
    if mode == 'train':
        #wandb: Define your config
        config = wandb.config
        config.epochs = n_epochs
        config.learning_rate = lr
        config.batch_size = BATCH_SIZE
        config.patience = PATIENCE

    if mode == 'train':
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            bert_model, device, clip=CLIP)
            # Log training loss to wandb
            if mode == 'train': wandb.log({"train_loss": np.asarray(loss).mean()})
            if x % 5 == 0: # We check the performance every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                                bert_model, lang, tokenizer, device)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                ot_precision = results_dev['ot_precision']
                ot_recall = results_dev['ot_recall']
                ot_f1 = results_dev['ot_f1']
                print(f'Validation Precision: {ot_precision} | Validation Recall: {ot_recall} | Validation Slot F1-score: {ot_f1}')
                
                
                if mode == 'train':
                    # Log validation loss to wandb
                    wandb.log({"val_loss": np.asarray(loss_dev).mean()})
                    wandb.log({"F1-score": ot_f1})
                    wandb.log({"ot_precision": ot_precision})
                    wandb.log({"ot_recall": ot_recall})
                
                # For decreasing the PATIENCE you can also use the average between slot f1 and intent accuracy
                if ot_f1 > best_f1:
                    best_f1 = ot_f1
                    # save best model!
                    model_info = {'state_dict': bert_model.state_dict(), 'lang':lang}
                    torch.save(model_info, model_path)
                    patience = PATIENCE
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    print("no more patience, finishing training")
                    break # Not nice but it keeps the code clean

    else:
        # Load model
        checkpoint = torch.load(model_path)
        lang = checkpoint['lang']
        bert_model = BertABSAModel(out_slot).to(device)
        bert_model.load_state_dict(checkpoint['model'])
        results_test, _ = eval_loop(test_loader, criterion_slots, 
                                            bert_model, lang, tokenizer, device)    
        ot_precision = results_test['ot_precision']
        ot_recall = results_test['ot_recall']
        ot_f1 = results_test['ot_f1']
        print(f'Test Precision: {ot_precision} | Test Recall: {ot_recall} | Test Slot F1-score: {ot_f1}')
        
