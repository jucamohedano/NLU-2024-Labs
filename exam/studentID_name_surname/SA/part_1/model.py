from transformers import BertModel
import torch.nn as nn


class BertABSAModel(nn.Module):
    def __init__(self, out_slot):
        super(BertABSAModel, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze BERT layers and replace top layers
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # Define output layers for multi-task learning
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, out_slot) # token-label classifcation head
    
    def forward(self, input_ids, attention_mask):
        slots, _ = self.bert(input_ids, attention_mask, return_dict=False) # sequence_output, pooled_output, (hidden_states), (attentions)

        slot_logits = self.slot_classifier(slots)
        return slot_logits
