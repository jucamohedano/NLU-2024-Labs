# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os
import math
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from model import *
from functions import *
from tqdm import tqdm
from functools import partial
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import DataLoader


def init_args():
    parser = argparse.ArgumentParser(description="Bert training for Aspect Term Extraction (ATE)")
    parser.add_argument("--mode", type=str, default='eval', help="Model mode: train or eval")
    parser.add_argument("--use_wandb", type=str, default='true', help="Use wandb to log training results.")
    parser.add_argument("--model_type", type=str, default='bert', help="Model to use.")
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

    parser = init_args()
    model_path = 'bin/best_model.pt'

    mode = parser.parse_args().mode
    use_wandb = parser.parse_args().use_wandb
    model_type = parser.parse_args().model_type
    if mode == 'train' and use_wandb == 'true':
        import wandb
        init_wandb()
    print(f'Running script in mode: {mode}. If you desire to change it use the --mode argument, i.e. python main.py --mode train')
    
    # load data
    train, val, test, vocab, _, ote_tag_vocab, _ = data_loader(parser)

    PAD_ID = vocab['PADDING']
    PUNCT_ID = vocab['PUNCT']
    slots = list(ote_tag_vocab.keys())

    lang = Lang(slots, pad_id=PAD_ID, punct_id=PUNCT_ID)
    out_slot = len(lang.slot2id)

    # Load BERT tokenizer
    tokenizr = None
    if model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ote_tags because we only want to predict the aspect, ts_tags also predict sentiment
    train_dataset = ATEDataset(train, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_id=PAD_ID, punct_id=PUNCT_ID)
    dev_dataset = ATEDataset(val, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_id=PAD_ID, punct_id=PUNCT_ID)
    test_dataset = ATEDataset(test, lang, tokenizer=tokenizer, tagging_scheme='ote_tags', pad_id=PAD_ID, punct_id=PUNCT_ID)

    # Dataloader instantiations
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_id=PAD_ID, device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_id=PAD_ID, device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_id=PAD_ID, device=device))
    
    # Create model
    model = None
    if model_type == 'roberta':
        model = RoBertaATEModel(out_slot).to(device)
    else:
        model = BertATEModel(out_slot).to(device)
    
    # hyperparams
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_loss_dev = math.inf
    n_epochs = 100
    CLIP = 5
    PATIENCE = 3
    patience = PATIENCE
    lr = 0.00001
    # betas = (0.9,0.99) # use default betas

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    config = None
    if mode == 'train' and use_wandb == 'true':
        #wandb: Define your config
        config = wandb.config
        config.epochs = n_epochs
        config.learning_rate = lr
        config.batch_size = BATCH_SIZE
        config.patience = PATIENCE
        # config.betas = betas

    if mode == 'train':
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            model, device, clip=CLIP)
            # Log training loss to wandb
            if use_wandb == 'true': wandb.log({"train_loss": np.asarray(loss).mean()})
            if x % 5 == 0: # We check the performance every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                                model, lang, tokenizer, device)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                ote_precision = results_dev['ot_precision']
                ote_recall = results_dev['ot_recall']
                ote_f1 = results_dev['ot_f1']
                print(f'Validation Precision: {ote_precision} | Validation Recall: {ote_recall} | Validation Slot F1-score: {ote_f1}')
                
                
                if use_wandb == 'true':
                    # Log validation loss to wandb
                    wandb.log({"val_loss": np.asarray(loss_dev).mean()})
                    wandb.log({"F1-score": ote_f1})
                    wandb.log({"ot_precision": ote_precision})
                    wandb.log({"ot_recall": ote_recall})
                
                # For decreasing the PATIENCE you can also use the average between slot f1 and intent accuracy
                if losses_dev[-1] > best_loss_dev:
                    best_loss_dev = losses_dev[-1]
                    # save best model!
                    model_info = {'state_dict': model.state_dict(), 'lang':lang}
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
        if model_type == 'roberta':
            model = RoBertaATEModel(out_slot).to(device)
        else:  
            model = BertATEModel(out_slot).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        results_test, _ = eval_loop(test_loader, criterion_slots, 
                                            model, lang, tokenizer, device)    
        ote_precision = results_test['ot_precision']
        ote_recall = results_test['ot_recall']
        ote_f1 = results_test['ot_f1']
        print(f'Test Precision: {ote_precision} | Test Recall: {ote_recall} | Test F1-score: {ote_f1}')
        
