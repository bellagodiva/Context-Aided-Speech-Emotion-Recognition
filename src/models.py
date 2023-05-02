import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from transformers import (AutoModel, AutoTokenizer, RobertaModel, Trainer, TrainingArguments, AutoConfig)



class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.out_dropout = hyp_params.out_dropout

        ####################################################################
        # Text EmoBERTa
        ####################################################################
        self.pretrained_model = "roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, use_fast=True)
        self.text_model = RobertaModel.from_pretrained(self.pretrained_model)
        
        # Projection layers (classifier)
        combined_dim = 1024
        output_dim = 4
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        
    def forward(self, x_l, x_l_mask):
        """
        text should have dimension [batch_size, seq_len, n_features]
        """
        ####################################################################
        #
        # Text RoBERTa
        #
        ####################################################################
        if self.training:
            self.text_model.train()
        else:
            self.text_model.eval()
        text_outputs = self.text_model(x_l, attention_mask = x_l_mask)
        last_h_t = text_outputs[0][:,0,:] #take cls (batch, 1 out of the whole seq len, roberta dim)
        last_h_t = torch.squeeze(last_h_t) #(batch, roberta dim)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_t)), p=self.out_dropout, training=self.training))
        output = self.out_layer(last_hs_proj)
        return output
