import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from transformers import (AutoModel, AutoTokenizer, RobertaModel, Trainer, TrainingArguments, AutoConfig)
import fairseq


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.d_a, self.d_v = 30, 30
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask


        #combined_dim = self.d_a + self.d_v
        self.partial_mode = self.aonly + self.vonly
        #self.proj_a = nn.Conv1d(74, 74, kernel_size=3, padding=1, bias=False)
        ckpt_path = "/mnt/hard2/bella/erc/hubert_large_ll60k.pt"
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.hubert_model = models[0]
        # Projection layers (classifier)
        combined_dim = 2048
        output_dim = 4
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        

        ####################################################################
        # Text EmoBERTa
        ####################################################################
        self.pretrained_model = "roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, use_fast=True)
        self.text_model = RobertaModel.from_pretrained(self.pretrained_model)
        embed_dim, attn_dropout = 74, self.attn_dropout
        self.trans_a_mem = TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, 3),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_l, audio_raw, x_l_mask, x_a_mask):
        """
        text and audio should have dimension [batch_size, seq_len, n_features]
        """
        ####################################################################
        #
        # Text EmoBERTa
        #
        ####################################################################
        if self.training:
            self.text_model.train()
        else:
            self.text_model.eval()
        text_outputs = self.text_model(x_l, attention_mask = x_l_mask)
        last_h_t = text_outputs[0][:,0,:] #take cls (batch, 1 out of the whole seq len, roberta dim)
        last_h_t = torch.squeeze(last_h_t) #(batch, roberta dim)
        if (len(last_h_t.shape)==1):
            last_h_t=last_h_t.unsqueeze(0)

        for param in self.hubert_model.feature_extractor.parameters():
            param.requires_grad = False
            
        inputs = {'source': audio_raw, 'padding_mask': ~x_a_mask}
        logits, padding_mask = self.hubert_model.extract_features(**inputs) #B,L,DIM
        #print(logits.shape)
        last_feat_pos = x_a_mask.sum(1)
        logits = logits.permute(1, 0, 2) #L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        a_logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        #print(a_logits.shape)
        #print(last_h_t.shape, a_logits.shape)
        last_hs = torch.cat([last_h_t, a_logits], dim=1)
        #last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_t)), p=self.out_dropout, training=self.training))
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        #last_hs_proj += last_h_t
        output = self.out_layer(last_hs_proj)
        return output
