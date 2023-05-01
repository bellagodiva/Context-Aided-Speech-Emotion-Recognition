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
        #self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        #self.orig_d_a, self.orig_d_v = hyp_params.orig_d_a, hyp_params.orig_d_v
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
        '''
        if self.partial_mode == 1:
            combined_dim = self.d_a   # assuming d_a == d_v
        else:
            combined_dim = 2 * self.d_a + 1024 #roberta out dim(768)
        
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)
        
        # 1. Temporal convolutional layers
        #self.proj_a = nn.Conv1d(74, self.d_a, kernel_size=3, padding=1, bias=False)
        #video (batch, seq len, 80, 80, 1) -> (batch, seq len, self.d_v)
        #CNN
        self.proj2d_v = nn.ModuleList([nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.ReLU(), 
                                       nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.ReLU(),
                                       #nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
                                       #nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2, stride=2)
                                    ])
        self.proj1d_v = nn.Conv1d(1600, self.d_v, kernel_size=3, padding=1, bias=False)
        #self.proj1d_v = nn.Conv1d(400, self.d_v, kernel_size=3, padding=1, bias=False)
        

        # 2. Crossmodal Attentions 
        if self.aonly:
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       '''
        #self.proj_a = nn.Conv1d(74, 74, kernel_size=3, padding=1, bias=False)
        self.proj_a = nn.Linear(74, 74)
        # Projection layers (classifier)
        combined_dim = 1098
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

    '''
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    '''       
    def forward(self, x_l, x_a, x_l_mask, x_a_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
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
        #text_outputs = self.text_model(x_l['input_ids'], attention_mask = x_l['attention_mask'])
        text_outputs = self.text_model(x_l, attention_mask = x_l_mask)
        last_h_t = text_outputs[0][:,0,:] #take cls (batch, 1 out of the whole seq len, roberta dim)
        last_h_t = torch.squeeze(last_h_t) #(batch, roberta dim)

        x_a = x_a.permute(1,0,2)
        a_logits = self.trans_a_mem(x_a)
        last_feat_pos = x_a_mask.sum(1)
        masks = torch.arange(a_logits.size(0), device=a_logits.device).expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        a_logits = (a_logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        #print(a_logits.shape)

        last_hs = torch.cat([last_h_t, a_logits], dim=1)



        '''

        ####################################################################
        #
        # Audio and Vision cross modal attention emo_capture
        #
        ####################################################################
        #bsz,seq_len,dim -> bsz,dim,seq_len
        x_a = x_a.transpose(1, 2)
        #x_v = x_v.transpose(1, 2)
       
        # Project the visual/audio features
        proj_x_a = self.proj_a(x_a)
        #print(proj_x_a.shape)
        #(bsz,length, H, W)-> (bsz*seq_len, H, W)
        bsz = x_v.shape[0]
        v_len = x_v.shape[1]
        x_v = x_v.view(bsz*v_len, x_v.shape[2], x_v.shape[3])
        #(bsz*seq_len, H, W) -> (bsz*seq_len, 1, H, W)
        x_v = x_v.unsqueeze(1)
        for layer in self.proj2d_v:
           x_v = layer(x_v)
        #(bsz*T,16(channel),5,5)-> (bsz,T,flatten)
        x_v = x_v.view(bsz,v_len,-1)
        #(bsz,T,flatten 400)->(bsz,400,T)
        x_v = x_v.transpose(1, 2)
        proj_x_v = self.proj1d_v(x_v)
        #(bsz,30,T) -> (T,bsz,30)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        #print(proj_x_v.shape, proj_x_a.shape)
        
        if self.aonly:
            # (L,V) --> A
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, key_padding_mask=~x_v_mask)
            h_as = self.trans_a_mem(h_a_with_vs, key_padding_mask=~x_a_mask)
            if type(h_as) == tuple:
                h_as = h_as[0]
            #last_h_a = last_hs = h_as[0]
            #take the last seq before padded elements
            last_h_a = last_hs = h_as[x_a_mask.sum(1)-1, torch.arange(h_as.size(1))]

        if self.vonly:
            # (L,A) --> V
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, key_padding_mask=~x_a_mask)
            h_vs = self.trans_v_mem(h_v_with_as, key_padding_mask=~x_v_mask)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            #take the last seq before padded elements
            last_h_v = last_hs = h_vs[x_v_mask.sum(1)-1, torch.arange(h_vs.size(1))]

        if self.partial_mode == 2:
            #print(last_h_a.shape, last_h_v.shape, last_h_t.shape )
            last_hs = torch.cat([last_h_a, last_h_v, last_h_t], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        #print('output:',output.shape, last_hs.shape)
        '''
        #last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_t)), p=self.out_dropout, training=self.training))
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        #last_hs_proj += last_h_t
        output = self.out_layer(last_hs_proj)
        return output