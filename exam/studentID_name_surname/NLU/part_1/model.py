import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Add biodirectionality to the LSTM layer. As a result the size of the hidden states is doubled
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        utt_emb = self.dropout(utt_emb) # we can use dropout after the embedding layer

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        #.cpu().numpy() converts the seq_lengths tensor to a NumPy array and moves it to the CPU memory
        # This is done because pack_padded_sequence expects the sequence lengths to be provided as a CPU-based NumPy array.
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        utt_emb = self.dropout(utt_encoded) # we can use dropout after the LSTM layer too!
        
        # Get the last [forward and backward] hidden states
        # Clarification: The last hidden state, obtained using last_hidden[-1, :, :], represents
        # the hidden state corresponding to the last time step of the sequences in the batch.
        # we concatenate the forward and backward hidden states along the hidden size dimension (dim=1)
        # last_hidden = last_hidden[-1,:,:] # without bidirectionality
        last_hidden = torch.cat((last_hidden[-2, :, :],  # last hidden state from the forward pass
                                 last_hidden[-1, :, :]), # last hidden state from the backward pass
                                 dim=1) # sequence dimension
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent


