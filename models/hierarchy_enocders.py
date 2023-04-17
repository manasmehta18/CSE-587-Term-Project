#!/usr/bin/env python
# -*- coding: utf-8 -*-'
import logging
import bcolz
import pickle
import os
from re import L
import re
import sys
from unicodedata import bidirectional
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.optim import Adam,AdamW
from torch import nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
# from torch._C import dtype, long
from torch.nn import LSTM, Linear, Dropout,ReLU, Tanh,Sequential,GRU,MaxPool1d,Embedding
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.nn.init import xavier_normal_,xavier_uniform_,uniform_,constant_,calculate_gain
from tqdm import utils

import transformers
from transformers import (
    # BertModel,
    BertTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# from utils.BERTModel import BertModel
from utils.modeling_bert import BertModel

# from transformers.utils.dummy_pt_objects import AdamW

sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
from utils.utils_cause_detection import InputFeatures, CauseDetectionDataset
from utils.tool_box import seed_torch

class HCT_test(nn.Module):
    def __init__(self, cause_encoder_path: str,  learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0):
        super().__init__()
        '''
        use bert to tag
        add speaker embedding
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)

        # default is Bi-LSTM & Attention
        # self.attention = Attention() if utterance_encoder_config["attention"] else None
        # del utterance_encoder_config["attention"]
        # self.utterance_encoder = LSTM(**utterance_encoder_config)
        # self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.relu = ReLU()
        self.tanh = Tanh()
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.linear = Sequential(Linear(in_features=768,out_features=768),
                                ReLU(),
                                Linear(in_features=768,out_features=out_dim))
        # self.linear = Sequential(Linear(in_features=1024,out_features=512),
        #                         ReLU(),
        #                         Linear(in_features=512,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        # self.utterance_encoder_config = utterance_encoder_config
        # self.cause_tagger_config = cause_tagger_config
        # self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        # self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.01}]  ## TODO: recover
        # params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        # self.optimizer = Adam(params,lr=learning_rate)
        small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        media_lr,media_param = learning_rate, ['utterance_encoder','cause_tagger']
        large_lr, large_param = 2e-3, ['linear','speaker_embed']
        params += [{"params":[p for n,p in named_params.items() if any([t in n and 'speaker_embed' not in n for t in small_param])],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        self.optimizer = AdamW(params)
        

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids,label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding 
        # token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        # utterance_hidden = token_hidden[:,0,:]  ## [turns_num, 768]  take [CLS] hidden state as utterance representation
        # # utterance_hidden=self.relu(utterance_hidden)
        # # utterance_hidden = self.drop_out(utterance_hidden)
        # # utterance encoding
        # all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        # padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        # packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        # tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        # contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        # assert len(target_cause_id) == len(turn_num)
        # index_list = []
        # accumulate_turn_num = 0
        # for i in range(len(target_cause_id)):
        #     ids = [id+accumulate_turn_num for id in target_cause_id[i]]
        #     index_list.extend(ids)
        #     accumulate_turn_num += longest_turn_num
        # indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        # out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        # tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        # tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        # bilinear_info=tmp_bilinear_info.permute(0,2,1).expand(expanded_dim,tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1])  ## [num_layers*num_directions, batch_size, direction*hidden_size_1]

        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)
        # last_hidden_state=self.relu(last_hidden_state)
        # tmp_last_hidden = last_hidden_state.permute(1,0,2)  ## [max_seq_length, batch_size, 768]
        t=last_hidden_state.shape[-1]
        tmp_last_hidden = last_hidden_state.view(-1,t)  ## [max_seq_length * batch_size, 768]
        output=self.linear(tmp_last_hidden)  
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden)[0] 
        # # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # # cause span tagging
        # input_dim=cause_info.shape[-1]
        # cause_info_seq = cause_info.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        # output=self.linear(cause_info_seq)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        max_step_num = input_ids.shape[1]
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_step_num

    def weight_init(self):
        # for n,p in self.utterance_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        # for n,p in self.cause_tagger.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.linear.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        all_speaker_ids = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_speaker_ids.append(f.spearker_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.extend(f.context_input_ids)
            all_context_attention_mask.extend(f.context_attention_mask)
            all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        speaker_ids = torch.tensor(
            all_speaker_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = torch.tensor(
            all_context_input_ids, dtype=torch.long).to(self.device)
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)

class HCT_test_pos(nn.Module):
    def __init__(self, cause_encoder_path: str,  learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0):
        super().__init__()
        '''
        use bert to tag
        add speaker embedding
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)

        # default is Bi-LSTM & Attention
        # self.attention = Attention() if utterance_encoder_config["attention"] else None
        # del utterance_encoder_config["attention"]
        # self.utterance_encoder = LSTM(**utterance_encoder_config)
        # self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.relu = ReLU()
        self.tanh = Tanh()
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.linear = Sequential(Linear(in_features=768,out_features=768),
                                ReLU(),
                                Linear(in_features=768,out_features=out_dim))
        # self.linear = Sequential(Linear(in_features=1024,out_features=512),
        #                         ReLU(),
        #                         Linear(in_features=512,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        # self.utterance_encoder_config = utterance_encoder_config
        # self.cause_tagger_config = cause_tagger_config
        # self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        # self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.01}]  ## TODO: recover
        # params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        # self.optimizer = Adam(params,lr=learning_rate)
        small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        media_lr,media_param = learning_rate, ['utterance_encoder','cause_tagger']
        large_lr, large_param = 1e-3, ['linear','speaker_embed']
        params += [{"params":[p for n,p in named_params.items() if any([t in n and 'speaker_embed' not in n for t in small_param])],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        self.optimizer = AdamW(params)
        

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, pos_ids,label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding 
        # token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        # utterance_hidden = token_hidden[:,0,:]  ## [turns_num, 768]  take [CLS] hidden state as utterance representation
        # # utterance_hidden=self.relu(utterance_hidden)
        # # utterance_hidden = self.drop_out(utterance_hidden)
        # # utterance encoding
        # all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        # padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        # packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        # tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        # contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        # assert len(target_cause_id) == len(turn_num)
        # index_list = []
        # accumulate_turn_num = 0
        # for i in range(len(target_cause_id)):
        #     ids = [id+accumulate_turn_num for id in target_cause_id[i]]
        #     index_list.extend(ids)
        #     accumulate_turn_num += longest_turn_num
        # indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        # out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        # tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        # tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        # bilinear_info=tmp_bilinear_info.permute(0,2,1).expand(expanded_dim,tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1])  ## [num_layers*num_directions, batch_size, direction*hidden_size_1]

        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids,pos_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)
        # last_hidden_state=self.relu(last_hidden_state)
        # tmp_last_hidden = last_hidden_state.permute(1,0,2)  ## [max_seq_length, batch_size, 768]
        t=last_hidden_state.shape[-1]
        tmp_last_hidden = last_hidden_state.view(-1,t)  ## [max_seq_length * batch_size, 768]
        output=self.linear(tmp_last_hidden)  
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden)[0] 
        # # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # # cause span tagging
        # input_dim=cause_info.shape[-1]
        # cause_info_seq = cause_info.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        # output=self.linear(cause_info_seq)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        max_step_num = input_ids.shape[1]
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_step_num

    def weight_init(self):
        # for n,p in self.utterance_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        # for n,p in self.cause_tagger.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.linear.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        all_pos_ids = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_pos_ids.append(f.position_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.extend(f.context_input_ids)
            all_context_attention_mask.extend(f.context_attention_mask)
            all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        pos_ids = torch.tensor(
            all_pos_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = torch.tensor(
            all_context_input_ids, dtype=torch.long).to(self.device)
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, pos_ids,label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)


class HCT_test_sabert(nn.Module):
    def __init__(self, cause_encoder_path: str,  learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0):
        super().__init__()
        '''
        use bert to tag
        add speaker embedding and dialogue structure (use SA-BERT)
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)

        ## add special token
        special_tokens_dict = {'additional_special_tokens': ['[EOU]','[EOT]','[EOE]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.cause_encoder.resize_token_embeddings(len(self.tokenizer))

        # default is Bi-LSTM & Attention
        # self.attention = Attention() if utterance_encoder_config["attention"] else None
        # del utterance_encoder_config["attention"]
        # self.utterance_encoder = LSTM(**utterance_encoder_config)
        # self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.relu = ReLU()
        self.tanh = Tanh()
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.linear = Sequential(Linear(in_features=768,out_features=768),
                                ReLU(),
                                Linear(in_features=768,out_features=out_dim))
        # self.linear = Sequential(Linear(in_features=1024,out_features=512),
        #                         ReLU(),
        #                         Linear(in_features=512,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        # self.utterance_encoder_config = utterance_encoder_config
        # self.cause_tagger_config = cause_tagger_config
        # self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        # self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.01}]  ## TODO: recover
        # params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        # self.optimizer = Adam(params,lr=learning_rate)
        small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        media_lr,media_param = learning_rate, ['utterance_encoder','cause_tagger']
        large_lr, large_param = 2e-3, ['linear','speaker_embed']
        params += [{"params":[p for n,p in named_params.items() if any([t in n and 'speaker_embed' not in n for t in small_param])],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        self.optimizer = AdamW(params)
        

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, speaker_ids,label_ids):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding 
        # token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        # utterance_hidden = token_hidden[:,0,:]  ## [turns_num, 768]  take [CLS] hidden state as utterance representation
        # # utterance_hidden=self.relu(utterance_hidden)
        # # utterance_hidden = self.drop_out(utterance_hidden)
        # # utterance encoding
        # all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        # padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        # packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        # tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        # contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        # assert len(target_cause_id) == len(turn_num)
        # index_list = []
        # accumulate_turn_num = 0
        # for i in range(len(target_cause_id)):
        #     ids = [id+accumulate_turn_num for id in target_cause_id[i]]
        #     index_list.extend(ids)
        #     accumulate_turn_num += longest_turn_num
        # indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        # out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        # tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        # tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        # bilinear_info=tmp_bilinear_info.permute(0,2,1).expand(expanded_dim,tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1])  ## [num_layers*num_directions, batch_size, direction*hidden_size_1]

        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids,speaker_ids=speaker_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)
        # last_hidden_state=self.relu(last_hidden_state)
        # tmp_last_hidden = last_hidden_state.permute(1,0,2)  ## [max_seq_length, batch_size, 768]
        t=last_hidden_state.shape[-1]
        tmp_last_hidden = last_hidden_state.view(-1,t)  ## [max_seq_length * batch_size, 768]
        output=self.linear(tmp_last_hidden)  
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden)[0] 
        # # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # # cause span tagging
        # input_dim=cause_info.shape[-1]
        # cause_info_seq = cause_info.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        # output=self.linear(cause_info_seq)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        max_step_num = input_ids.shape[1]
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_step_num

    def weight_init(self):
        # for n,p in self.utterance_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        # for n,p in self.cause_tagger.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.linear.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        all_speaker_ids = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_speaker_ids.append(f.spearker_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.extend(f.context_input_ids)
            all_context_attention_mask.extend(f.context_attention_mask)
            all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        speaker_ids = torch.tensor(
            all_speaker_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        # context_input_ids = torch.tensor(
        #     all_context_input_ids, dtype=torch.long).to(self.device)
        # context_attention_mask = torch.tensor(
        #     all_context_attention_mask, dtype=torch.float).to(self.device)
        # context_token_type_ids = torch.tensor(
        #     all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        # assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, speaker_ids,label_ids)


class HCT_test_com(nn.Module):
    def __init__(self, token_enocder_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0):
        super().__init__()
        '''
        use bert to tag
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)

        # default is Bi-LSTM & Attention
        self.attention = Attention() if utterance_encoder_config["attention"] else None
        del utterance_encoder_config["attention"]
        # self.utterance_encoder = LSTM(**utterance_encoder_config)
        # self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.relu = ReLU()
        self.tanh = Tanh()
        self.drop_out = Dropout(p=dropout)
        self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        in_dim = num_directions * cause_tagger_config['hidden_size']
        media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.linear = Sequential(Linear(in_features=768,out_features=768),
                                ReLU(),
                                Linear(in_features=768,out_features=out_dim))
        # self.linear = Sequential(Linear(in_features=1024,out_features=512),
        #                         ReLU(),
        #                         Linear(in_features=512,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.01}]  ## TODO: recover
        # params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        # self.optimizer = Adam(params,lr=learning_rate)
        small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        media_lr,media_param = learning_rate, ['utterance_encoder','cause_tagger']
        large_lr, large_param = 2e-3, ['linear']
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        self.optimizer = AdamW(params)
        
        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, label_ids, target_cause_id):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)
        # last_hidden_state=self.relu(last_hidden_state)
        # tmp_last_hidden = last_hidden_state.permute(1,0,2)  ## [max_seq_length, batch_size, 768]
        t=last_hidden_state.shape[-1]
        tmp_last_hidden = last_hidden_state.view(-1,t)  ## [max_seq_length * batch_size, 768]
        output=self.linear(tmp_last_hidden)  
        # calculate loss
        max_step_num = input_ids.shape[1]
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_step_num

    def weight_init(self):
        # for n,p in self.utterance_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        # for n,p in self.cause_tagger.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.linear.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            # all_context_input_ids.extend(f.context_input_ids)
            # all_context_attention_mask.extend(f.context_attention_mask)
            # all_context_token_type_ids.extend(f.context_token_type_ids)
            # turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = torch.tensor(
            all_context_input_ids, dtype=torch.long).to(self.device)
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, target_cause_id)

class HCT_test_both(nn.Module):
    def __init__(self, embed_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0,max_seq_len:int=512,max_seq_context:int=128,over_writte:bool=False):
        super().__init__()
        '''
        GRU as cause tagger & and context encoder
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        root_dir = embed_path.rsplit(".",1)[0]+".dat"
        out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
        out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
        if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
            ## process and cache glove ===========================================
            words = []
            idx = 0
            word2idx = {}    
            vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
            with open(os.path.join(embed_path),"rb") as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)
            vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=root_dir, mode='w')
            vectors.flush()
            pickle.dump(words, open(out_dir_word, 'wb'))
            pickle.dump(word2idx, open(out_dir_idx, 'wb'))
            print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
            ## =======================================================
        ## load glove
        print("load Golve from {}".format(embed_path.rsplit("/",1)[0]))
        vectors = bcolz.open(root_dir)[:]
        words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
        self.word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))

        weights_matrix = np.zeros((400002, 300))  ## unk & pad  ## default fix
        weights_matrix[1] = np.random.normal(scale=0.6, size=(300, ))
        weights_matrix[2:,:] = vectors
        weights_matrix = torch.FloatTensor(weights_matrix)

        pad_idx,unk_idx = 0,1
        self.embed = Embedding(400002, 300,padding_idx=pad_idx)  
        # self.embed.load_state_dict({'weight': weights_matrix})
        self.embed.from_pretrained(weights_matrix,freeze=False,padding_idx=pad_idx)
        
        self.max_seq_len = max_seq_len
        self.max_seq_context = max_seq_context
        self.embed_dim = weights_matrix.shape[-1]
        # self.token_encoder = GRU(300,150,bidirectional=True)
        # self.max_pool = MaxPool1d(128,128)
        # self.map_1 = Linear(300,300)
        # self.tanh = Tanh()
        # self.cause_encoder = BertModel.from_pretrained(
        #     cause_encoder_path)  ## default is span-bert-base
        # self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)
        self.cause_tagger = LSTM(300,300,num_layers=2,bidirectional=True)

        # default is Bi-LSTM & Attention
        # self.attention = Attention() if utterance_encoder_config["attention"] else None
        # del utterance_encoder_config["attention"]
        # self.utterance_encoder = LSTM(300,150,bidirectional=True)
        # self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        # self.map_2 = Linear(300,768)
        # self.relu = ReLU()
        
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.projector = Sequential(Linear(600,300),
                                ReLU(),
                                Linear(300,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.1}]  ## TODO: recover
        params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        self.optimizer = AdamW(params,lr=learning_rate)
        # small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        # media_lr,media_param = 2e-4, ['utterance_encoder','cause_tagger']
        # large_lr, large_param = 2e-3, ['linear']
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        # self.optimizer = AdamW(params)

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding
        max_seq_len =  input_ids.shape[1]
        # context_input_embed = []
        # for ids in context_input_ids:
        #     context_input_embed.append(self.embed(ids))
        # context_input=torch.cat(context_input_embed,dim=0).permute(1,0,2)   ## [128,turns_num,300]
        # context_input=pad_sequence(context_input_ids).view(-1,self.max_seq_embed,self.embed_dim)  ## [longest_turn_num * batch_size, 128,300]
        # packed_context_input= pack_padded_sequence(context_input,turn_num,enforce_sorted=False)
        # token_hidden=self.token_encoder(context_input)[0].permute(1,2,0)  ## [96,300,128]
        # # token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        # tmp_utterance_hidden = self.max_pool(token_hidden).squeeze(-1) ## [turns_num, 300] 
        # utterance_hidden = self.tanh(self.map_1(tmp_utterance_hidden)) ## [turns_num, 300]
        # utterance_hidden=self.relu(utterance_hidden)
        # utterance_hidden = self.drop_out(utterance_hidden)
        # utterance encoding
        # all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        # padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        # packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        # tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        # longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        # contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        # assert len(target_cause_id) == len(turn_num)
        # index_list = []
        # accumulate_turn_num = 0
        # for i in range(len(target_cause_id)):
        #     ids = [id+accumulate_turn_num for id in target_cause_id[i]]
        #     index_list.extend(ids)
        #     accumulate_turn_num += longest_turn_num
        # indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        # out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        # tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        # tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        # batch_size,bilinear_dim=tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1]
        # # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        # bilinear_info=tmp_bilinear_info.permute(2,0,1).expand(batch_size,max_seq_len,bilinear_dim)  ## [batch_size,128,300]

        # cause encoding
        input_rep = self.embed(input_ids)  ## [batch_size, max_seq_length, 300]
        input_rep = self.drop_out(input_rep)
        pack_input = pack_padded_sequence(input_rep.permute(1,0,2),lengths=token_type_ids,enforce_sorted=False)
        last_hidden_state_tmp = self.cause_tagger(pack_input)[0]  ## [max_seq_length, batch_size, 600]
        last_hidden_state,_=pad_packed_sequence(last_hidden_state_tmp,batch_first=True,total_length=max_seq_len)  ## [ batch_size, max_seq_length, 600]
        out_rep = self.drop_out(last_hidden_state)  
        # context_info = self.drop_out(bilinear_info)
        # tmp_last_hidden = torch.cat((last_hidden_state,context_info),dim=2)  ## [batch_size, 128, 1068]
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # cause span tagging
        # input_dim=tmp_last_hidden.shape[-1]
        # cause_info_seq = tmp_last_hidden.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        output=self.projector(out_rep).view(-1,3)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_seq_len

    # can save the model via model.state_dict() directly, because the bert used here is bare version (no need to save config)
    # def save(self,save_path:str):
    #     token_enocder_path = os.path.join(save_path,"token_encoder")
    #     os.makedirs(token_enocder_path,exist_ok=True)
    #     model_to_save = (self.token_encoder.module if hasattr(self.token_encoder, "module") else self.token_encoder)
    #     model_to_save.save_pretrained(token_enocder_path)
    #     self.tokenizer.save_pretrained(token_enocder_path)

    #     cause_encoder_path = os.path.join(save_path,"cause_encoder")
    #     os.makedirs(cause_encoder_path,exist_ok=True)
    #     model_to_save = (self.cause_encoder.module if hasattr(self.cause_encoder, "module") else self.cause_encoder)
    #     model_to_save.save_pretrained(cause_encoder_path)
    #     self.tokenizer.save_pretrained(cause_encoder_path)

    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    # def load(self):
    #     pass
    def weight_init(self):
        # for n,p in self.utterance_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        # for n,p in self.token_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.cause_tagger.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.projector.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        # for n,p in self.map_1.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_uniform_(p,gain=calculate_gain('relu'))
        #         elif "bias" in n:
        #             constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.extend(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.append(torch.tensor(f.context_input_ids,dtype=torch.int).to(self.device))
            # all_context_attention_mask.extend(f.context_attention_mask)
            # all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.int).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = all_token_type_ids
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = all_context_input_ids
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        # assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)

class HCT_test_both_2(nn.Module):
    def __init__(self, embed_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0,max_seq_len:int=512,max_seq_context:int=128,over_writte:bool=False):
        super().__init__()
        '''
        GRU as cause tagger & and context encoder
        add context info
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        root_dir = embed_path.rsplit(".",1)[0]+".dat"
        out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
        out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
        if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
            ## process and cache glove ===========================================
            words = []
            idx = 0
            word2idx = {}    
            vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
            with open(os.path.join(embed_path),"rb") as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)
            vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=root_dir, mode='w')
            vectors.flush()
            pickle.dump(words, open(out_dir_word, 'wb'))
            pickle.dump(word2idx, open(out_dir_idx, 'wb'))
            print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
            ## =======================================================
        ## load glove
        print("load Golve from {}".format(embed_path.rsplit("/",1)[0]))
        vectors = bcolz.open(root_dir)[:]
        words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
        self.word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))

        weights_matrix = np.zeros((400002, 300))  ## unk & pad  ## default fix
        weights_matrix[1] = np.random.normal(scale=0.6, size=(300, ))
        weights_matrix[2:,:] = vectors
        weights_matrix = torch.FloatTensor(weights_matrix)

        pad_idx,unk_idx = 0,1
        self.embed_1 = Embedding(400002, 300,padding_idx=pad_idx)  
        # self.embed.load_state_dict({'weight': weights_matrix})
        self.embed_1.from_pretrained(weights_matrix,freeze=False,padding_idx=pad_idx)

        self.embed_2 = Embedding(400002, 300,padding_idx=pad_idx)  
        # self.embed.load_state_dict({'weight': weights_matrix})
        self.embed_2.from_pretrained(weights_matrix,freeze=False,padding_idx=pad_idx)
        
        self.max_seq_len = max_seq_len
        self.max_seq_context = max_seq_context
        self.embed_dim = weights_matrix.shape[-1]
        self.token_encoder = GRU(300,150,bidirectional=True)
        self.max_pool = MaxPool1d(128,128)
        self.map_1 = Linear(300,300)
        self.tanh = Tanh()
        # self.cause_encoder = BertModel.from_pretrained(
        #     cause_encoder_path)  ## default is span-bert-base
        # self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)
        self.cause_tagger = LSTM(300,300,num_layers=2,bidirectional=True)

        # default is Bi-LSTM & Attention
        self.attention = Attention() if utterance_encoder_config["attention"] else None
        del utterance_encoder_config["attention"]
        self.utterance_encoder = LSTM(300,150,bidirectional=True)
        self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.map_2 = Linear(300,300)
        self.relu = ReLU()
        
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.projector = Sequential(Linear(900,512),
                                ReLU(),
                                Linear(512,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.1}]  ## TODO: recover
        params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        self.optimizer = AdamW(params,lr=learning_rate)
        # small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        # media_lr,media_param = 2e-4, ['utterance_encoder','cause_tagger']
        # large_lr, large_param = 2e-3, ['linear']
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        # self.optimizer = AdamW(params)

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding
        max_seq_len =  input_ids.shape[1]
        context_input_embed = []
        for ids in context_input_ids:
            context_input_embed.append(self.embed_1(ids))
        context_input=torch.cat(context_input_embed,dim=0).permute(1,0,2)   ## [128,turns_num,300]
        # context_input=pad_sequence(context_input_ids).view(-1,self.max_seq_embed,self.embed_dim)  ## [longest_turn_num * batch_size, 128,300]
        # packed_context_input= pack_padded_sequence(context_input,turn_num,enforce_sorted=False)
        token_hidden=self.token_encoder(context_input)[0].permute(1,2,0)  ## [96,300,128]
        # token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        tmp_utterance_hidden = self.max_pool(token_hidden).squeeze(-1) ## [turns_num, 300] 
        utterance_hidden = self.tanh(self.map_1(tmp_utterance_hidden)) ## [turns_num, 300]
        # utterance_hidden=self.relu(utterance_hidden)
        # utterance_hidden = self.drop_out(utterance_hidden)
        # utterance encoding
        all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        assert len(target_cause_id) == len(turn_num)
        index_list = []
        accumulate_turn_num = 0
        for i in range(len(target_cause_id)):
            ids = [id+accumulate_turn_num for id in target_cause_id[i]]
            index_list.extend(ids)
            accumulate_turn_num += longest_turn_num
        indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        batch_size,bilinear_dim=tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1]
        # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        bilinear_info=tmp_bilinear_info.permute(2,0,1).expand(batch_size,max_seq_len,bilinear_dim)  ## [batch_size,128,300]

        # cause encoding
        input_rep = self.embed_2(input_ids)  ## [batch_size, max_seq_length, 300]
        input_rep = self.drop_out(input_rep)
        pack_input = pack_padded_sequence(input_rep.permute(1,0,2),lengths=token_type_ids,enforce_sorted=False)
        last_hidden_state_tmp = self.cause_tagger(pack_input)[0]  ## [max_seq_length, batch_size, 600]
        last_hidden_state,_=pad_packed_sequence(last_hidden_state_tmp,batch_first=True,total_length=max_seq_len)  ## [ batch_size, max_seq_length, 600]
        out_rep = self.drop_out(last_hidden_state)  
        context_info = self.drop_out(bilinear_info)
        # context_info = bilinear_info
        tmp_last_hidden = torch.cat((out_rep,context_info),dim=2)  ## [batch_size, 512,900]
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # cause span tagging
        # input_dim=tmp_last_hidden.shape[-1]
        # cause_info_seq = tmp_last_hidden.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        # output=self.projector(out_rep).view(-1,3)  ## [max_seq_length * batch_size, num_label]
        output=self.projector(tmp_last_hidden).view(-1,3)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_seq_len

    # can save the model via model.state_dict() directly, because the bert used here is bare version (no need to save config)
    # def save(self,save_path:str):
    #     token_enocder_path = os.path.join(save_path,"token_encoder")
    #     os.makedirs(token_enocder_path,exist_ok=True)
    #     model_to_save = (self.token_encoder.module if hasattr(self.token_encoder, "module") else self.token_encoder)
    #     model_to_save.save_pretrained(token_enocder_path)
    #     self.tokenizer.save_pretrained(token_enocder_path)

    #     cause_encoder_path = os.path.join(save_path,"cause_encoder")
    #     os.makedirs(cause_encoder_path,exist_ok=True)
    #     model_to_save = (self.cause_encoder.module if hasattr(self.cause_encoder, "module") else self.cause_encoder)
    #     model_to_save.save_pretrained(cause_encoder_path)
    #     self.tokenizer.save_pretrained(cause_encoder_path)

    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    # def load(self):
    #     pass
    def weight_init(self):
        for n,p in self.utterance_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.token_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.cause_tagger.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.projector.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.map_1.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.extend(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.append(torch.tensor(f.context_input_ids,dtype=torch.int).to(self.device))
            # all_context_attention_mask.extend(f.context_attention_mask)
            # all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.int).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = all_token_type_ids
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = all_context_input_ids
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        # assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)

class HCT_cat(nn.Module):
    def __init__(self, token_enocder_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0):
        super().__init__()
        '''
        hierarchy cause tagging model, which is utilized to tag the cause span in the given utterances
        '''
        self.token_encoder = BertModel.from_pretrained(
            token_enocder_path)  ## default is bert=base=cased
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(token_enocder_path)

        # default is Bi-LSTM & Attention
        self.attention = Attention() if utterance_encoder_config["attention"] else None
        del utterance_encoder_config["attention"]
        self.utterance_encoder = LSTM(**utterance_encoder_config)
        self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.relu = ReLU()
        self.tanh = Tanh()
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        in_dim = num_directions * cause_tagger_config['hidden_size']
        media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        self.linear = Linear(in_features=300,out_features=out_dim)
        self.map = Linear(768,300)
        # self.linear = Sequential(Linear(in_features=300,out_features=300),
        #                         ReLU(),
        #                         Linear(in_features=300,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.01}]  ## TODO: recover
        # params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        # self.optimizer = Adam(params,lr=learning_rate)
        small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        media_lr,media_param = learning_rate, ['utterance_encoder','cause_tagger']
        large_lr, large_param = learning_rate, ['linear']
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        self.optimizer = AdamW(params)
        

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding 
        token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        utterance_hidden = token_hidden[:,0,:]  ## [turns_num, 768]  take [CLS] hidden state as utterance representation
        # utterance_hidden=self.relu(utterance_hidden)
        # utterance_hidden = self.drop_out(utterance_hidden)
        # utterance encoding
        all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        assert len(target_cause_id) == len(turn_num)
        index_list = []
        accumulate_turn_num = 0
        for i in range(len(target_cause_id)):
            ids = [id+accumulate_turn_num for id in target_cause_id[i]]
            index_list.extend(ids)
            accumulate_turn_num += longest_turn_num
        indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        # bilinear_info=tmp_bilinear_info.permute(0,2,1).expand(expanded_dim,tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1])  ## [num_layers*num_directions, batch_size, direction*hidden_size_1]
        
        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)
        cause_rep=self.map(last_hidden_state)
        batch_size,max_seq_len = cause_rep.shape[0],cause_rep.shape[1]
        bilinear_info = tmp_bilinear_info.squeeze(0).contiguous().permute(1,0).unsqueeze(1).expand(batch_size,max_seq_len,300)  ## [batch_size, max_seq_length, direction*hidden_size_1]
        tmp_cat=torch.stack((cause_rep,bilinear_info),dim=3).contiguous().view(-1,300,2)
        rep_dim  = bilinear_info.shape[-1]
        cause_info_seq=self.mean_pool(tmp_cat).squeeze()  ## [batch_size * max_seq_length, direction*hidden_size_1]

        # last_hidden_state=self.relu(last_hidden_state)
        # tmp_last_hidden = last_hidden_state.permute(1,0,2)  ## [max_seq_length, batch_size, 768]
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # cause span tagging
        # input_dim=cause_info.shape[-1]
        # cause_info_seq = cause_info.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        output=self.linear(cause_info_seq)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_seq_len

    def weight_init(self):
        for n,p in self.utterance_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        # for n,p in self.cause_tagger.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.linear.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.extend(f.context_input_ids)
            all_context_attention_mask.extend(f.context_attention_mask)
            all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = torch.tensor(
            all_context_input_ids, dtype=torch.long).to(self.device)
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)

class HCT_mod(nn.Module):
    def __init__(self, embed_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0,max_seq_embed:int=128,over_writte:bool=False):
        super().__init__()
        '''
        GRU as token encoder
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        root_dir = embed_path.rsplit(".",1)[0]+".dat"
        out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
        out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
        if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
            ## process and cache glove ===========================================
            words = []
            idx = 0
            word2idx = {}    
            vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
            with open(os.path.join(embed_path),"rb") as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)
            vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=root_dir, mode='w')
            vectors.flush()
            pickle.dump(words, open(out_dir_word, 'wb'))
            pickle.dump(word2idx, open(out_dir_idx, 'wb'))
            print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
            ## =======================================================
        ## load glove
        print("load Golve from {}".format(embed_path.rsplit("/",1)[0]))
        vectors = bcolz.open(root_dir)[:]
        words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
        self.word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))

        weights_matrix = np.zeros((400002, 300))  ## unk & pad  ## default fix
        weights_matrix[1] = np.random.normal(scale=0.6, size=(300, ))
        weights_matrix[2:,:] = vectors
        weights_matrix = torch.FloatTensor(weights_matrix)

        pad_idx,unk_idx = 0,1
        self.embed = Embedding(400002, 300,padding_idx=pad_idx)  
        # self.embed.load_state_dict({'weight': weights_matrix})
        self.embed.from_pretrained(weights_matrix,freeze=False,padding_idx=pad_idx)
        
        self.max_seq_embed = max_seq_embed
        self.embed_dim = weights_matrix.shape[-1]
        self.token_encoder = GRU(300,150,bidirectional=True)
        self.max_pool = MaxPool1d(128,128)
        self.map_1 = Linear(300,300)
        self.tanh = Tanh()
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)

        # default is Bi-LSTM & Attention
        # self.attention = Attention() if utterance_encoder_config["attention"] else None
        # del utterance_encoder_config["attention"]
        self.utterance_encoder = LSTM(300,150,bidirectional=True)
        self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        # self.map_2 = Linear(300,768)
        # self.relu = ReLU()
        
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.projector = Sequential(Linear(300+768,300),
                                ReLU(),
                                Linear(300,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.1}]  ## TODO: recover
        params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        self.optimizer = AdamW(params,lr=learning_rate)
        # small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        # media_lr,media_param = 2e-4, ['utterance_encoder','cause_tagger']
        # large_lr, large_param = 2e-3, ['linear']
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        # self.optimizer = AdamW(params)

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding
        max_seq_len =  input_ids.shape[1]
        context_input_embed = []
        for ids in context_input_ids:
            context_input_embed.append(self.embed(ids))
        context_input=torch.cat(context_input_embed,dim=0).permute(1,0,2)   ## [128,turns_num,300]
        # context_input=pad_sequence(context_input_ids).view(-1,self.max_seq_embed,self.embed_dim)  ## [longest_turn_num * batch_size, 128,300]
        # packed_context_input = pack_padded_sequence(context_input,turn_num,enforce_sorted=False)
        token_hidden=self.token_encoder(context_input)[0].permute(1,2,0)  ## [96,300,128]
        # token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        tmp_utterance_hidden = self.max_pool(token_hidden).squeeze(-1) ## [turns_num, 300] 
        utterance_hidden = self.tanh(self.map_1(tmp_utterance_hidden)) ## [turns_num, 300]
        # utterance_hidden=self.relu(utterance_hidden)
        # utterance_hidden = self.drop_out(utterance_hidden)
        # utterance encoding
        all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        assert len(target_cause_id) == len(turn_num)
        index_list = []
        accumulate_turn_num = 0
        for i in range(len(target_cause_id)):
            ids = [id+accumulate_turn_num for id in target_cause_id[i]]
            index_list.extend(ids)
            accumulate_turn_num += longest_turn_num
        indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        batch_size,bilinear_dim=tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1]
        # expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        bilinear_info=tmp_bilinear_info.permute(2,0,1).expand(batch_size,max_seq_len,bilinear_dim)  ## [batch_size,128,300]

        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)  ## [batch_size, max_seq_length, 768]
        context_info = self.drop_out(bilinear_info)
        # last_hidden_state=self.relu(last_hidden_state)
        tmp_last_hidden = torch.cat((last_hidden_state,context_info),dim=2)  ## [batch_size, 128, 1068]
        # seq_step = []
        # for i in range(attention_mask.shape[0]):
        #     ins_att=attention_mask[i,:].detach().to("cpu").tolist()
        #     try:
        #         step_length = ins_att.index(0)
        #     except ValueError:
        #         step_length = len(ins_att)
        #     seq_step.append(step_length)
        # max_step_num = max(seq_step)
        # packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        # cell_init = torch.zeros_like(bilinear_info)
        # cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        # cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # cause span tagging
        input_dim=tmp_last_hidden.shape[-1]
        cause_info_seq = tmp_last_hidden.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        output=self.projector(cause_info_seq)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_seq_len

    # can save the model via model.state_dict() directly, because the bert used here is bare version (no need to save config)
    # def save(self,save_path:str):
    #     token_enocder_path = os.path.join(save_path,"token_encoder")
    #     os.makedirs(token_enocder_path,exist_ok=True)
    #     model_to_save = (self.token_encoder.module if hasattr(self.token_encoder, "module") else self.token_encoder)
    #     model_to_save.save_pretrained(token_enocder_path)
    #     self.tokenizer.save_pretrained(token_enocder_path)

    #     cause_encoder_path = os.path.join(save_path,"cause_encoder")
    #     os.makedirs(cause_encoder_path,exist_ok=True)
    #     model_to_save = (self.cause_encoder.module if hasattr(self.cause_encoder, "module") else self.cause_encoder)
    #     model_to_save.save_pretrained(cause_encoder_path)
    #     self.tokenizer.save_pretrained(cause_encoder_path)

    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    # def load(self):
    #     pass
    def weight_init(self):
        for n,p in self.utterance_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.token_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        # for n,p in self.cause_tagger.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.projector.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.map_1.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.append(torch.tensor(f.context_input_ids,dtype=torch.int).to(self.device))
            # all_context_attention_mask.extend(f.context_attention_mask)
            # all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = all_context_input_ids
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        # assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)

class HCT(nn.Module):
    def __init__(self, token_enocder_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0):
        super().__init__()
        '''
        hierarchy cause tagging model, which is utilized to tag the cause span in the given utterances
        '''
        self.token_encoder = BertModel.from_pretrained(
            token_enocder_path)  ## default is bert=base=cased
        self.cause_encoder = BertModel.from_pretrained(
            cause_encoder_path)  ## default is span-bert-base
        self.tokenizer = BertTokenizer.from_pretrained(token_enocder_path)

        # default is Bi-LSTM & Attention
        self.attention = Attention() if utterance_encoder_config["attention"] else None
        del utterance_encoder_config["attention"]
        self.utterance_encoder = LSTM(**utterance_encoder_config)
        self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        self.relu = ReLU()
        self.tanh = Tanh()
        self.drop_out = Dropout(p=dropout)
        self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        in_dim = num_directions * cause_tagger_config['hidden_size']
        media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.linear = Sequential(Linear(in_features=in_dim,out_features=media_dim),
                                ReLU(),
                                Linear(in_features=media_dim,out_features=out_dim))

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.01}]  ## TODO: recover
        # params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        # self.optimizer = Adam(params,lr=learning_rate)
        small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        media_lr,media_param = 2e-4, ['utterance_encoder','cause_tagger']
        large_lr, large_param = 2e-3, ['linear']
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        self.optimizer = AdamW(params)
        

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding 
        token_hidden=self.token_encoder(context_input_ids,context_attention_mask,context_token_type_ids)[0]  ## [turns_num, max_seq_length, 768]
        utterance_hidden = token_hidden[:,0,:]  ## [turns_num, 768]  take [CLS] hidden state as utterance representation
        # utterance_hidden=self.relu(utterance_hidden)
        # utterance_hidden = self.drop_out(utterance_hidden)
        # utterance encoding
        all_dialog_hidden=torch.split(utterance_hidden,turn_num)
        padded_sequence=pad_sequence([dialog for dialog in all_dialog_hidden])  ## [longest_turn_num, batch_size, 768]
        packed_sequence = pack_padded_sequence(padded_sequence,lengths=turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
        tmp_contextual_out=self.utterance_encoder(packed_sequence)[0]  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        tmp_contextual_out, _ = pad_packed_sequence(tmp_contextual_out,batch_first=False)  ## [longest_turn_num, batch_size, direction*hidden_size_1]
        longest_turn_num, tmp_dim = tmp_contextual_out.shape[0], tmp_contextual_out.shape[-1]
        contextual_out = tmp_contextual_out.view(-1,tmp_dim)  ## [longest_turn_num * batch_size, direction*hidden_size_1]
        assert len(target_cause_id) == len(turn_num)
        index_list = []
        accumulate_turn_num = 0
        for i in range(len(target_cause_id)):
            ids = [id+accumulate_turn_num for id in target_cause_id[i]]
            index_list.extend(ids)
            accumulate_turn_num += longest_turn_num
        indices = torch.tensor(index_list,dtype=torch.int).to(self.device)
        out=torch.index_select(contextual_out,0,indices)  ## [batch_size*2, direction*hidden_size_1]
        tmp_out=out.permute(1,0).unsqueeze(0)  ## [1, direction*hidden_size_1, batch_size*2]
        tmp_bilinear_info=self.mean_pool(tmp_out)  ## [1, direction*hidden_size_1, batch_size]
        expanded_dim = self.cause_tagger_config["num_layers"] * self.cause_tagger_config["num_directions"]
        bilinear_info=tmp_bilinear_info.permute(0,2,1).expand(expanded_dim,tmp_bilinear_info.shape[2],tmp_bilinear_info.shape[1])  ## [num_layers*num_directions, batch_size, direction*hidden_size_1]

        # cause encoding
        last_hidden_state = self.cause_encoder(input_ids,attention_mask,token_type_ids)[0]  ## [batch_size, max_seq_length, 768]
        last_hidden_state = self.drop_out(last_hidden_state)
        # last_hidden_state=self.relu(last_hidden_state)
        tmp_last_hidden = last_hidden_state.permute(1,0,2)  ## [max_seq_length, batch_size, 768]
        seq_step = []
        for i in range(attention_mask.shape[0]):
            ins_att=attention_mask[i,:].detach().to("cpu").tolist()
            try:
                step_length = ins_att.index(0)
            except ValueError:
                step_length = len(ins_att)
            seq_step.append(step_length)
        max_step_num = max(seq_step)
        packed_hidden = pack_padded_sequence(tmp_last_hidden,seq_step,batch_first=False,enforce_sorted=False)
        cell_init = torch.zeros_like(bilinear_info)
        cause_info_pack=self.cause_tagger(packed_hidden,(bilinear_info,cell_init))[0]  ## [max_seq_length, batch_size, hidden_size_2]  ## assert hidden_size_2 == direction*hidden_size_1
        cause_info, _ = pad_packed_sequence(cause_info_pack,batch_first=False)  ## [longest_step_num, batch_size, hidden_size_2]

        # cause span tagging
        input_dim=cause_info.shape[-1]
        cause_info_seq = cause_info.view(-1,input_dim)  ## [max_seq_length * batch_size, hidden_size_2]
        output=self.linear(cause_info_seq)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        # labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_step_num

    # can save the model via model.state_dict() directly, because the bert used here is bare version (no need to save config)
    # def save(self,save_path:str):
    #     token_enocder_path = os.path.join(save_path,"token_encoder")
    #     os.makedirs(token_enocder_path,exist_ok=True)
    #     model_to_save = (self.token_encoder.module if hasattr(self.token_encoder, "module") else self.token_encoder)
    #     model_to_save.save_pretrained(token_enocder_path)
    #     self.tokenizer.save_pretrained(token_enocder_path)

    #     cause_encoder_path = os.path.join(save_path,"cause_encoder")
    #     os.makedirs(cause_encoder_path,exist_ok=True)
    #     model_to_save = (self.cause_encoder.module if hasattr(self.cause_encoder, "module") else self.cause_encoder)
    #     model_to_save.save_pretrained(cause_encoder_path)
    #     self.tokenizer.save_pretrained(cause_encoder_path)

    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    #     torch.save({'state_dict': self.utterance_encoder.state_dict()}, os.path.join(save_path, "utterance_encoder.pth.tar"))
    # def load(self):
    #     pass
    def weight_init(self):
        for n,p in self.utterance_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.cause_tagger.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.linear.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_attention_mask.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.extend(f.context_input_ids)
            all_context_attention_mask.extend(f.context_attention_mask)
            all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = torch.tensor(
            all_token_type_ids, dtype=torch.long).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = torch.tensor(
            all_context_input_ids, dtype=torch.long).to(self.device)
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == attention_mask.shape == token_type_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label_ids, context_input_ids, context_attention_mask, context_token_type_ids, target_cause_id,turn_num)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward():
        pass


class HCT_test_intention(nn.Module):
    def __init__(self, embed_path: str, cause_encoder_path: str, utterance_encoder_config: dict, cause_tagger_config: dict, learning_rate:float=5e-5, 
                cuda:int=0, seed:int=42,dropout:float=0.0,max_seq_len:int=512,max_seq_context:int=128,over_writte:bool=False,intention:bool=False,interaction:bool=False):
        super().__init__()
        '''
        GRU as cause tagger & and context encoder
        use intention and interaction info 
        '''
        # self.token_encoder = BertModel.from_pretrained(
        #     token_enocder_path)  ## default is bert=base=cased
        root_dir = embed_path.rsplit(".",1)[0]+".dat"
        out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
        out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
        if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
            ## process and cache glove ===========================================
            words = []
            idx = 0
            word2idx = {}    
            vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
            with open(os.path.join(embed_path),"rb") as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)
            vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=root_dir, mode='w')
            vectors.flush()
            pickle.dump(words, open(out_dir_word, 'wb'))
            pickle.dump(word2idx, open(out_dir_idx, 'wb'))
            print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
            ## =======================================================
        ## load glove
        print("load Golve from {}".format(embed_path.rsplit("/",1)[0]))
        vectors = bcolz.open(root_dir)[:]
        words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
        self.word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))

        weights_matrix = np.zeros((400002, 300))  ## unk & pad  ## default fix
        weights_matrix[1] = np.random.normal(scale=0.6, size=(300, ))
        weights_matrix[2:,:] = vectors
        weights_matrix = torch.FloatTensor(weights_matrix)

        self.intention = intention
        self.interaction = interaction

        pad_idx,unk_idx = 0,1
        self.embed = Embedding(400002, 300,padding_idx=pad_idx)  
        # self.embed.load_state_dict({'weight': weights_matrix})
        self.embed.from_pretrained(weights_matrix,freeze=False,padding_idx=pad_idx)
        
        self.max_seq_len = max_seq_len
        self.max_seq_context = max_seq_context
        self.embed_dim = weights_matrix.shape[-1]
        self.token_encoder = GRU(300,150,bidirectional=True)
        self.max_pool_1 = MaxPool1d(self.max_seq_len,self.max_seq_len)
        self.max_pool_2 = MaxPool1d(self.max_seq_context,self.max_seq_context)
        self.map_2 = Linear(300,300) if self.intention else None
        self.map_3 = Linear(300,300) if self.interaction else None
        self.tanh = Tanh()
        # self.max_pool = MaxPool1d(128,128)
        # self.map_1 = Linear(300,300)
        # self.tanh = Tanh()
        # self.cause_encoder = BertModel.from_pretrained(
        #     cause_encoder_path)  ## default is span-bert-base
        # self.tokenizer = BertTokenizer.from_pretrained(cause_encoder_path)
        self.num_layer = 2
        self.bidirectional = True
        self.cause_tagger = LSTM(300,300,num_layers=self.num_layer,bidirectional=self.bidirectional)

        # default is Bi-LSTM & Attention
        # self.attention = Attention() if utterance_encoder_config["attention"] else None
        # del utterance_encoder_config["attention"]
        # self.utterance_encoder = LSTM(300,150,bidirectional=True)
        # self.mean_pool = nn.AvgPool1d(2, stride=2)  ## mean pool the target&cause representation
        # self.map_2 = Linear(300,768)
        # self.relu = ReLU()
        
        self.drop_out = Dropout(p=dropout)
        # self.cause_tagger = LSTM(**cause_tagger_config)  ## default is Uni-LSTM

        # num_directions = 2 if cause_tagger_config["bidirectional"] else 1
        # in_dim = num_directions * cause_tagger_config['hidden_size']
        # media_dim = in_dim
        out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        # self.linear = Linear(in_dim,out_dim)
        self.projector = Sequential(Linear(600,300),
                                ReLU(),
                                Linear(300,out_features=out_dim))

        # === intention ===
        self.intention_encoder = LSTM(300,150,num_layers=1,bidirectional=True) if intention else None
        # =================

        # === interaction ===
        self.interaction_encoder = LSTM(300,150,num_layers=1,bidirectional=True) if interaction else None
        # =================

        self.criterion = nn.CrossEntropyLoss()
        self._ignore_id = self.criterion.ignore_index

        self.utterance_encoder_config = utterance_encoder_config
        self.cause_tagger_config = cause_tagger_config
        self.utterance_encoder_config["num_directions"] = 2 if self.utterance_encoder_config["bidirectional"] else 1
        self.cause_tagger_config["num_directions"] = 2 if self.cause_tagger_config["bidirectional"] else 1

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params += [{"params":[p for n,p in named_params.items() if not any([nd in n for nd in no_decay])],"weight_decay":0.1}]  ## TODO: recover
        params += [{"params":[p for n, p in named_params.items() if any(nd in n for nd in no_decay)],"weight_decay":0.0}]
        self.optimizer = AdamW(params,lr=learning_rate)
        # small_lr,small_param = learning_rate, ['token_encoder','cause_encoder']
        # media_lr,media_param = 2e-4, ['utterance_encoder','cause_tagger']
        # large_lr, large_param = 2e-3, ['linear']
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param])],"lr":small_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param])],"lr":media_lr}]
        # params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param])],"lr":large_lr}]
        # self.optimizer = AdamW(params)

        if torch.cuda.is_available():
            if cuda == -1:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda:{}".format(cuda))
        else:
            if cuda != -1:
                warnings.warn("no gpu is available, switch to cpu")
            self.device = "cpu"

        seed_torch(seed)

    def forward(self,input_ids, token_type_ids, label_ids, all_speaker_intention,all_real_time_inter,spk_turn_num,itr_turn_num):
        ## TODO: add activate func & drop
        ## TODO: pad input rep of cause tagger lstm
        # token encoding
        max_seq_len =  input_ids.shape[1]
        batch_size = input_ids.shape[0]
        direction = 2 if self.bidirectional else 1
        expand_dim = direction * self.num_layer
        ## intention 
        cell_h = None
        if self.intention:
            in_ids_spk=torch.cat(all_speaker_intention,dim=0).permute(1,0)  ## [128,turns_num]
            input_spk=self.embed(in_ids_spk)  ## [128,turns_num,300]
            token_hidden_spk=self.token_encoder(input_spk)[0].permute(1,2,0)  ## [96,300,128]
            tmp_utterance_hidden_spk = self.max_pool_2(token_hidden_spk).squeeze(-1) ## [turns_num, 300] 
            utterance_hidden_spk = self.tanh(self.map_2(tmp_utterance_hidden_spk)) ## [turns_num, 300]
            all_hidden_spk=torch.split(utterance_hidden_spk,spk_turn_num)
            padded_sequence_spk=pad_sequence([it for it in all_hidden_spk])  ## [longest_turn_num, batch_size, 768]
            packed_sequence_spk = pack_padded_sequence(padded_sequence_spk,lengths=spk_turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
            spk_intention=self.intention_encoder(packed_sequence_spk)[0]
            spk_intention=pad_packed_sequence(spk_intention,batch_first=True)[0]  ## [batch_size, longest_turn_num, 300]
            all_intention = []
            for i in range(batch_size):
                all_intention.append(torch.mean(spk_intention[i,:,:],dim=0))  ## [300,]
                # all_intention.append(torch.max(spk_intention[i,:,:],dim=0)[0])  ## [300,]
            all_intention=torch.stack(all_intention,dim=0)  ## [batch_size,300]
            cell_h = all_intention.unsqueeze(0)  ## [1,batch_size,300]
            cell_h = cell_h.repeat(expand_dim,1,1)
        hidden_h = None
        if self.interaction:
            in_ids_rt=torch.cat(all_real_time_inter,dim=0).permute(1,0)  ## [128,turns_num]
            input_rt=self.embed(in_ids_rt)  ## [128,turns_num,300]
            token_hidden_rt=self.token_encoder(input_rt)[0].permute(1,2,0)  ## [96,300,128]
            tmp_utterance_hidden_rt = self.max_pool_2(token_hidden_rt).squeeze(-1) ## [turns_num, 300] 
            utterance_hidden_rt = self.tanh(self.map_3(tmp_utterance_hidden_rt)) ## [turns_num, 300]
            all_hidden_rt=torch.split(utterance_hidden_rt,itr_turn_num)
            padded_sequence_rt=pad_sequence([rt for rt in all_hidden_rt])  ## [longest_turn_num, batch_size, 768]
            packed_sequence_rt = pack_padded_sequence(padded_sequence_rt,lengths=itr_turn_num,batch_first=False,enforce_sorted=False)  ## so the LSTM will not care the padded step
            rt_interaction=self.interaction_encoder(packed_sequence_rt)[0]
            rt_interaction=pad_packed_sequence(rt_interaction,batch_first=True)[0]  ## [batch_size, longest_turn_num, 300]
            all_interaction = []
            for i in range(batch_size):
                all_interaction.append(torch.mean(rt_interaction[i,:,:],dim=0))  ## [300,]
                # all_interaction.append(torch.max(rt_interaction[i,:,:],dim=0)[0])  ## [300,]
            all_interaction=torch.stack(all_interaction,dim=0)  ## [batch_size,300]
            hidden_h = all_interaction.unsqueeze(0)
            hidden_h = hidden_h.repeat(expand_dim,1,1)
        if hidden_h is not None and cell_h is not None:
            context_info = (hidden_h,cell_h)
        elif hidden_h is not None:
            context_info = (hidden_h,torch.zeros_like(hidden_h))
        elif cell_h is not None:
            context_info = (torch.zeros_like(cell_h),cell_h)
        else:
            context_info = None

        # cause encoding
        input_rep = self.embed(input_ids)  ## [batch_size, max_seq_length, 300]
        input_rep = self.drop_out(input_rep)
        pack_input = pack_padded_sequence(input_rep.permute(1,0,2),lengths=token_type_ids,enforce_sorted=False)
        if context_info is None:
            last_hidden_state_tmp = self.cause_tagger(pack_input)[0]  ## [max_seq_length, batch_size, 600]
        else:
            last_hidden_state_tmp = self.cause_tagger(pack_input,context_info)[0]  ## [max_seq_length, batch_size, 600]
        last_hidden_state,_=pad_packed_sequence(last_hidden_state_tmp,batch_first=True,total_length=max_seq_len)  ## [ batch_size, max_seq_length, 600]
        out_rep = self.drop_out(last_hidden_state)  
        # out_rep = last_hidden_state
        output=self.projector(out_rep).view(-1,3)  ## [max_seq_length * batch_size, num_label]

        # calculate loss
        labels = label_ids.view(-1)  ## [max_seq_length * batch_size]
        # labels=label_ids[:,0:max_step_num].contiguous().view(-1)  ## [longest_step_num * batch_size]
        loss = self.criterion(output,labels)

        return loss, output, labels, max_seq_len

    def weight_init(self):
        # for n,p in self.utterance_encoder.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_normal_(p)
        #         elif "bias" in n:
        #             constant_(p,0)
        for n,p in self.token_encoder.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.cause_tagger.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_normal_(p)
                elif "bias" in n:
                    constant_(p,0)
        for n,p in self.projector.named_parameters():
            if p.requires_grad:
                if 'weight' in n:
                    xavier_uniform_(p,gain=calculate_gain('relu'))
                elif "bias" in n:
                    constant_(p,0)
        # for n,p in self.map_1.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_uniform_(p,gain=calculate_gain('relu'))
        #         elif "bias" in n:
        #             constant_(p,0)
        if self.intention:
            for n,p in self.intention_encoder.named_parameters():
                if p.requires_grad:
                    if 'weight' in n:
                        xavier_normal_(p)
                    elif "bias" in n:
                        constant_(p,0)
            for n,p in self.map_2.named_parameters():
                if p.requires_grad:
                    if 'weight' in n:
                        xavier_uniform_(p,gain=calculate_gain('relu'))
                    elif "bias" in n:
                        constant_(p,0)
        if self.interaction:
            for n,p in self.interaction_encoder.named_parameters():
                if p.requires_grad:
                    if 'weight' in n:
                        xavier_normal_(p)
                    elif "bias" in n:
                        constant_(p,0)
            for n,p in self.map_3.named_parameters():
                if p.requires_grad:
                    if 'weight' in n:
                        xavier_uniform_(p,gain=calculate_gain('relu'))
                    elif "bias" in n:
                        constant_(p,0)
        # for n,p in self.map_1.named_parameters():
        #     if p.requires_grad:
        #         if 'weight' in n:
        #             xavier_uniform_(p,gain=calculate_gain('relu'))
        #         elif "bias" in n:
        #             constant_(p,0)
        
    def feature2input(self,batch_features: List[InputFeatures]):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        This include:
        1. cat the context (which is special in cause detection)
        2. transfer to tensor
        3. to device
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], [], [], [], []
        turn_num = []
        spk_turn_num = []
        itr_turn_num = []
        all_speaker_intention = []
        all_real_time_inter = []
        for f in batch_features:
            all_input_ids.append(f.input_ids)
            all_token_type_ids.extend(f.token_type_ids)
            all_label_ids.append(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            # target_id.append(f.target_id)
            real_time_num = f.target_id + 1
            all_real_time_inter.append(torch.tensor(f.context_input_ids[:real_time_num],dtype=torch.int).to(self.device))
            itr_turn_num.append(real_time_num)
            # speaker_id.append(torch.tensor(f.spearker_type_ids,dtype=torch.int).to(self.device))
            speaker_ids = f.spearker_ids
            assert len(speaker_ids) == len(f.context_input_ids)
            context = np.array(f.context_input_ids)
            speaker_intention = None
            for id in list(set(speaker_ids)):
                if id == f.current_speaker:
                    idex= [idx for idx,tt in enumerate(speaker_ids) if tt == id]
                    speaker_intention = torch.tensor(context[idex],dtype=torch.int).to(self.device)
            assert speaker_intention is not None
            all_speaker_intention.append(speaker_intention)
            spk_turn_num.append(speaker_intention.shape[0])
            all_context_input_ids.append(torch.tensor(f.context_input_ids,dtype=torch.int).to(self.device))
            # all_context_attention_mask.extend(f.context_attention_mask)
            # all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
        input_ids = torch.tensor(
            all_input_ids, dtype=torch.int).to(self.device)
        attention_mask = torch.tensor(
            all_attention_mask, dtype=torch.float).to(self.device)
        token_type_ids = all_token_type_ids
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        # context_input_ids = all_context_input_ids
        # context_attention_mask = torch.tensor(
        #     all_context_attention_mask, dtype=torch.float).to(self.device)
        # context_token_type_ids = torch.tensor(
        #     all_context_token_type_ids, dtype=torch.long).to(self.device)

        assert input_ids.shape == label_ids.shape  ## [batch_size, max_seq_length]
        # assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]

        return (input_ids,token_type_ids, label_ids, all_speaker_intention,all_real_time_inter,spk_turn_num,itr_turn_num)

if __name__ == "__main__":
    input_ids = [1, 2, 3, 4, 5]
    attention_mask = [1, 1, 1, 0, 0]
    token_type_ids = [0, 0, 0, 0, 0]
    label_ids = [1, 1, 1, 1, 1]
    context_input_ids = [[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]
    context_attention_mask = [[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]
    context_token_type_ids = [[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]]

    input_feature = InputFeatures(input_ids, attention_mask, token_type_ids,
                                  label_ids, context_token_type_ids, context_input_ids, context_attention_mask)

    data_path = "/data/lourenze/corpus/RECCON_IOB/alpha_IOB/new_dailydialog_test.json"
    context_path = "/data/lourenze/corpus/RECCON_IOB/alpha_IOB/dailydialog_test_shared_dialog.json"
    labels = ["B-cause", "I-cause", "O"]
    tokenizer_path = "/data/lourenze/PretrainedLM/spanbert-base-cased"
    # feature = convert_raw_data_to_features(
    #     data_path=data_path, context_path=context_path, label_list=label_list, max_seq_length=128, tokenizer=tokenizer)
    data_set = CauseDetectionDataset(data_path=data_path, context_path=context_path, labels=labels,
                                     max_seq_length=128, tokenizer_path=tokenizer_path, model_type="bert")
    token_enocder_path = "/data/lourenze/PretrainedLM/bert-base-cased"
    cause_encoder_path = "/data/lourenze/PretrainedLM/spanbert-base-cased"
    utterance_encoder_config = {"input_size":768,"hidden_size":150,"num_layers":1,"dropout":0.2,"bidirectional":True,"attention":False}
    cause_tagger_config = {"input_size":768,"hidden_size":300,"num_layers":1,"dropout":0.2,"bidirectional":False}
    # test_model = HCT(token_enocder_path=token_enocder_path, cause_encoder_path=cause_encoder_path, utterance_encoder_config=utterance_encoder_config,
    #  cause_tagger_config=cause_tagger_config,cuda=2)
    # test_model.to(test_model.device)
    # torch.save(test_model.state_dict(),"./model.pt")
    # test_model.load_state_dict(torch.load("./model.pt"))
    for start in range(0, len(data_set), 2):
        batch_features = data_set[start:start+2]
        # batch_input=test_model.feature2input(batch_features)
        # loss,output,labels=test_model(*batch_input)
    #     # cv = test_model.feature2input
    #     # batch_input = cv(batch_features)
    #     batch_input=test_model.feature2input(batch_features)
    #     print(batch_input)


