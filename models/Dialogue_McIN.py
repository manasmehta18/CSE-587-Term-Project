#!/usr/bin/env python
# -*- coding: utf-8 -*-'
import logging
import bcolz
import pickle
import os
import numpy as np
import torch
import re
import sys
import warnings

from re import L
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
from unicodedata import bidirectional
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, sequence_labeling
from torch.nn.modules.activation import ELU
from torch.optim import Adam,AdamW
from torch import nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.nn import LSTM, Linear, Dropout,ReLU, Tanh,Sequential,GRU,MaxPool1d,Embedding
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.nn.init import xavier_normal_,xavier_uniform_,uniform_,constant_,calculate_gain
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.graph_conv import GraphConv

sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))

import transformers
from transformers import (
    BertModel,
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

from utils.cause_detection import InputFeatures
from utils.tool_box import bert_find_max_len,word_embedding,xavier_norm

def emotion_embedding(embed_path:str,idx2emo:dict):
    _, weights_matrix, word2idx, emo_dim = word_embedding(embed_path,over_writte=False)
    ini_vector = np.zeros((len(idx2emo), emo_dim)) 
    for idx,emo in idx2emo.items():
        idx_ = word2idx.get(emo,None)
        assert idx_ is not None
        emo_feature = weights_matrix[idx_,:]
        ini_vector[idx] = emo_feature
    emo_embed = Embedding(len(idx2emo), emo_dim)  
    emo_embed.from_pretrained(torch.FloatTensor(ini_vector),freeze=False)

    return emo_embed,emo_dim

def speaker_embedding(speaker_num:int=2,spk_dim:int=100):
    ini_vector = np.random.normal(scale=0.6, size=(speaker_num, spk_dim))
    spk_embed = Embedding(speaker_num, spk_dim)  
    spk_embed.from_pretrained(torch.FloatTensor(ini_vector),freeze=False)

    return spk_embed

def split_ins_id(ins_id:str):
    ori_ins_id,cause_ids = ins_id.split("_cause_utt_")[0],ins_id.split("_cause_utt_")[1]
    cause_ids = cause_ids.split("_")
    new_ins_id = [ori_ins_id + "_cause_utt_" + cause_id for cause_id in cause_ids]

    return new_ins_id

class GraphNetwork(nn.Module):
    def __init__(self,node_dim,hidden_dim,aggr="mean",heads=1,concat=True):
        super().__init__()
        '''
        apply GNNs (RGCN + GAT) on the relation-aware multi-causal graph

        pre-defined relation type:
        - cause-cause:
            0. contiguous_past
            1. contiguous_future
            2. distant_past
            3. distant_future
        - cause-target:
            4. in_context
            5. contiguous
            6. distant
        - emotion-utterance:
            7. affect
        '''
        self.n_relation = 8
        self.conv1 = RGCNConv(node_dim,hidden_dim,self.n_relation,num_bases=30,aggr=aggr)
        self.conv2 = GATConv(hidden_dim,hidden_dim,heads=heads,concat=concat,dropout=0.0,add_self_loops=True)
        self.conv3 = GraphConv(hidden_dim,hidden_dim,aggr="add")  ## useless
    
    def forward(self,target_node,cause_node,emotion_node,word_node,target_idx, cause_idx):
        '''
        target_node -> [batch_size, 600]
        emotion_node -> [batch_size, 600]
        cause_node -> [batch_cause_num, 600]
        word_node -> [batch_cause_num, 512, 600]
        '''
        batch_size,node_dim = target_node.shape[0],target_node.shape[1]
        batch_cause_num,max_seq_len = word_node.shape[0],word_node.shape[1]
        x,edge_index,edge_type,directed_edge_num = self.construct_relation_graph(target_node,cause_node,emotion_node,word_node,target_idx, cause_idx)
        out_1 = self.conv1(x,edge_index,edge_type)
        out_2 = self.conv2(out_1,edge_index)  ## [2*batch_size+batch_cause_num, 600]
        
        cause_utt_node = out_2[2*batch_size:,:]  ## [batch_cause_num, 600]
        # concat directly
        out_final = cause_utt_node.unsqueeze(1).expand(batch_cause_num,max_seq_len,node_dim)  ## [batch_cause_num, 512, 600]

        return out_final, (out_1,out_2)

    def cause_target_rel(self,tgt_turn_id,cs_turn_ids):
        ''' len(edge_tp) == len(cs_turn_ids) '''
        edge_tp = []
        for cs_turn_id in cs_turn_ids:
            delta=abs(tgt_turn_id-cs_turn_id)
            if delta == 0:
                ## in_context
                edge_tp.append(4)
            elif delta == 1:
                ## contiguous
                edge_tp.append(5)
            else:
                ## distant
                edge_tp.append(6)

        return edge_tp
    
    def cause_cause_rel(self,current_cs_id,other_cs_id):
        ''' len(edge_tp) == len(other_cs_id) '''
        edge_tp = []
        for cs_id in other_cs_id:
            delta = abs(current_cs_id-cs_id)
            future = current_cs_id < cs_id
            if delta == 1:
                ## contiguous
                if future:
                    ### future
                    edge_tp.append(1)
                else:
                    ### past
                    edge_tp.append(0)
            elif delta > 1:
                ## distant
                if future:
                    ### future
                    edge_tp.append(3)
                else:
                    ### past
                    edge_tp.append(2)
            else:
                raise RuntimeError("invalid cause utterance id, delta == 0")

        return edge_tp

    def generate_rel(self, tgt_id,emo_id,cs_id,tgt_turn_id,cs_turn_id):
        ''' generate relational edges for a graph '''
        st_lis,ed_lis,type_lis = [],[],[]
        ## emotion-utterance 
        st_lis.extend([emo_id]*(len(cs_id)+1))
        ed_lis.extend([tgt_id]+cs_id)
        type_lis.extend([7]*(len(cs_id)+1))
        ### bidirection
        st_lis.extend([tgt_id]+cs_id)
        ed_lis.extend([emo_id]*(len(cs_id)+1))
        type_lis.extend([7]*(len(cs_id)+1))
        ## cause-target
        tgt_cs_tp = self.cause_target_rel(tgt_turn_id,cs_turn_id)
        st_lis.extend([tgt_id]*len(cs_id))
        ed_lis.extend(cs_id)
        type_lis.extend(tgt_cs_tp)
        ### bidirection
        st_lis.extend(cs_id)
        ed_lis.extend([tgt_id]*len(cs_id))
        type_lis.extend(tgt_cs_tp)
        ## cause-cause
        for p, id in enumerate(cs_id):
            current_turn_id=cs_turn_id[p]  
            st_lis.extend([id]*(len(cs_id)-1))
            tmp_cs_id = [t for t in cs_id if t != id]
            other_turn_id = [cs_turn_id[i] for i,t in enumerate(cs_id) if t != id]
            ed_lis.extend(tmp_cs_id)
            inter_cs_tp = self.cause_cause_rel(current_turn_id,other_turn_id)
            type_lis.extend(inter_cs_tp)
        
        assert len(st_lis) == len(ed_lis) == len(type_lis)

        return st_lis,ed_lis,type_lis 

    def construct_relation_graph(self,target_node,cause_node,emotion_node,word_node,target_idx, cause_idx):
        batch_size = target_node.shape[0]  ## graph_num
        cause_num_per_graph = [len(id) for id in cause_idx]
        assert len(cause_num_per_graph) == batch_size and sum(cause_num_per_graph) == word_node.shape[0]
        assert len(target_idx) == len(cause_idx) == batch_size
        x = torch.cat((target_node,emotion_node,cause_node),dim=0)  ## [all_node_num, 600]  ## all_node_num == 2 * batch_size + batch_cause_num
        edge_type = []
        start,end = [],[]
        cause_num_all = 0
        for id,cause_num in enumerate(cause_num_per_graph):
            # generate relational edges for each graph in a mini-batch
            emo_id, tgt_id = id + batch_size, id
            cs_id = [t+2*batch_size+cause_num_all for t in range(cause_num)]
            cause_num_all += cause_num
            tgt_turn_id,cs_turn_id = target_idx[id],cause_idx[id]
            st_lis, ed_lis, type_lis = self.generate_rel(tgt_id,emo_id,cs_id,tgt_turn_id,cs_turn_id)
            start.extend(st_lis)
            end.extend(ed_lis)
            edge_type.extend(type_lis)
        directed_edge_num = len(start)
        edge_index = torch.tensor([start,end],dtype=torch.long,device=x.device)  ## [2, edge_num]
        edge_type = torch.tensor(edge_type,dtype=torch.long,device=x.device)  ## [edge_num]

        return x, edge_index, edge_type, directed_edge_num

    def construct_word_graph(self,cause_utt_node,word_node,word_mask):
        assert cause_utt_node.shape[0] == word_node.shape[0]
        node_dim,max_seq_len,batch_cause_num = word_node.shape[-1],word_node.shape[-2],word_node.shape[0]
        word_node = word_node.view(-1,node_dim)  ## [batch_cause_num*max_seq_len, node_dim]
        word_mask = word_mask.view(-1)  ## [batch_cause_num*max_seq_len]
        word_mask_np = word_mask.cpu().numpy()
        true_word_index = np.argwhere(word_mask_np == 1).squeeze()  ## one-dim np.array indicating the true index of word nodes
        x = torch.cat((word_node,cause_utt_node),dim=0)  ## [batch_cause_num*max_seq_len+batch_cause_num, node_dim]
        word_num = batch_cause_num * max_seq_len
        start,end = [],[]
        for i in range(batch_cause_num):
            st,ed = i * max_seq_len, (i+1) * max_seq_len
            cause_id = i + word_num
            current_ins_true = (true_word_index >= st).astype(np.int) * (true_word_index < ed).astype(np.int)
            true_num = np.sum(current_ins_true)
            tmp_lis=current_ins_true.tolist()
            idx_st = tmp_lis.index(1)
            tmp_lis.reverse()
            idx_ed = len(tmp_lis)-tmp_lis.index(1)
            ed_idx = true_word_index[idx_st:idx_ed]
            assert len(ed_idx) == true_num
            start.extend([cause_id]*true_num)
            end.extend(ed_idx)
        assert len(start) == len(end)
        directed_edge_num = len(start)
        edge_index = torch.tensor([start,end],dtype=torch.long,device=x.device)  ## [2, edge_num]

        return x,edge_index,directed_edge_num


class CausalGCN(nn.Module):
    def __init__(self, token_enocder_path: str,lstm_in_dim:int=300,lstm_out_dim:int=150,bidirectional:bool=True,spk_dim:int=100,
                node_dim:int=600,activation:str="tanh",small_lr:float=2e-5,media_lr:float=2e-4,large_lr:float=1e-3,
                embed_path:str=None,emotion_dic:dict=None,cuda:int=0,seed:int=44,spk_embed:bool=False,dropout:float=0.5,weight_decay:float=0.0):
        super().__init__()
        '''
        token encdoer -> spanbert
        context encoder -> Bi-LSTM
        cause detectioner -> RGCN (w/o relation-attention) + GAT
        projector -> linear + tanh + dropout
        projector(cat contextual_info and speaker_embed) -> utt_node
        projector(spanbert_hidden) -> word_node
        projector(emotional_embed) -> emotion_node
        causalGCN(word_node,emotion_node,utt_node)

        cat spanbert_hidden and GCN output_hidden 
        '''
        self.token_encoder = BertModel.from_pretrained(token_enocder_path)  ## spanbert
        self.tokenizer = BertTokenizer.from_pretrained(token_enocder_path)
        self.context_encoder = LSTM(lstm_in_dim,lstm_out_dim,num_layers=1,bidirectional=bidirectional)
        self.graph_network = GraphNetwork(node_dim,hidden_dim=node_dim)

        self.relu = ReLU()
        if activation=="tanh":
            self.act = Tanh() 
        elif activation=="relu":
            self.act = ReLU()
        else:
            self.act = None        

        self.contex_dim = 2 * lstm_out_dim if bidirectional else lstm_out_dim
        self.turn_dim = self.contex_dim + spk_dim if spk_embed else self.contex_dim
        self.node_dim = node_dim

        self.idx2emo = emotion_dic
        self.emo2idx = dict([(v,k) for k,v in emotion_dic.items()])
        self.emo_embed,emo_dim = emotion_embedding(embed_path,self.idx2emo)  ## use pre_trained embedding by default
        self.spk_embed = speaker_embedding(2,spk_dim) if spk_embed else None
        
        self.dp_proj = Dropout(dropout)

        self.utt_proj = Sequential(Linear(768,lstm_in_dim),self.act,self.dp_proj) if self.act else Sequential(Linear(768,lstm_in_dim),self.dp_proj)
        self.cause_proj = Sequential(Linear(self.turn_dim,self.node_dim),self.act,self.dp_proj) if self.act else Sequential(Linear(self.turn_dim,self.node_dim),self.dp_proj)
        self.target_proj = Sequential(Linear(self.turn_dim,self.node_dim),self.act,self.dp_proj) if self.act else Sequential(Linear(self.turn_dim,self.node_dim),self.dp_proj)
        self.emo_proj = Sequential(Linear(emo_dim,self.node_dim),self.act,self.dp_proj) if self.act else Sequential(Linear(emo_dim,self.node_dim),self.dp_proj)
        self.word_proj = Sequential(Linear(768,self.node_dim),self.act,self.dp_proj) if self.act else Sequential(Linear(768,self.node_dim),self.dp_proj)

        self.out_dim = 3  ## default {"O","I-cause","B-cause"} formulation
        self.classifier = Sequential(Linear(768+self.node_dim,self.node_dim),  ## cat input and output of GraphNetwork
                                    self.relu,
                                    Linear(self.node_dim,self.out_dim))

        self.criterion = nn.CrossEntropyLoss() 
        self._ignore_id = self.criterion.ignore_index

        named_params = {name:param for name,param in self.named_parameters() if param.requires_grad == True}
        params = []
        small_param = ['token_encoder']
        media_param = ['graph_network']
        large_param_decay = ['context_encoder','_proj','classifier']
        large_param_no_decay = ['spk_embed','emo_embed']
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in small_param]) and p.requires_grad],"lr":small_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in media_param]) and p.requires_grad],"lr":media_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param_no_decay]) and p.requires_grad],"lr":large_lr}]
        params += [{"params":[p for n,p in named_params.items() if any([t in n for t in large_param_decay]) and p.requires_grad],"lr":large_lr,"weight_decay":weight_decay}]
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

    def forward(self,input_ids, attention_mask, token_type_ids, turn_num,target_idx, cause_idx,target_spk_idx,cause_spk_idx,emotion_lbs,label_ids,ins_id_lis):
        '''
        input_ids.shape == attention_mask.shape == token_type_ids.shape == [all_turn_num, max_seq_length], concat all the utt. of a mini-batch
        len(turn_num) == batch_size, dialog turn num of each instance in a mini-batch
        len(target_idx) == len(cause_idx), the turn idx of target utt. and cause utt. for each instance
        target_spk_idx.shape == [batch_size], cause_spk_idx.shape == [batch_cause_num],
        emotion_lbs.shape == [batch_size]
        label_ids.shape == [batch_cause_num, max_seq_length]
        '''
        batch_size = len(turn_num)
        max_step_num = input_ids.shape[1]
        # token encoding
        last_hidden_state = self.token_encoder(input_ids,attention_mask,token_type_ids)[0]  ## [all_turn_num, max_seq_length, 768]
        word_hidden = last_hidden_state[:,:,:]  ## [all_turn_num, 512, 768]
        cls_hidden = last_hidden_state[:,0,:]  ## [all_turn_num, 768]
        utt_rep = self.utt_proj(cls_hidden)  ## [all_turn_num, 300]
        batch_dialog = torch.split(utt_rep,turn_num,dim=0)  ## len(batch_dialog) == batch_size
        batch_word_hidden = torch.split(word_hidden,turn_num,dim=0)
        batch_dialog = pad_sequence(batch_dialog)  ## [longest_turn_num, batch_size, 300]
        packed_dialog = pack_padded_sequence(batch_dialog,turn_num,enforce_sorted=False)
        contextual_rep = self.context_encoder(packed_dialog)[0]
        contextual_rep = pad_packed_sequence(contextual_rep,batch_first=True)[0]  ## [batch_size, longest_turn_num, 300]
        batch_target,batch_cause = [],[]
        batch_cause_word = []
        for i,tgt_idx in enumerate(target_idx):
            cs_idx = cause_idx[i]
            batch_target.append(contextual_rep[i,tgt_idx,:].unsqueeze(0))
            batch_cause.append(contextual_rep[i,cs_idx,:])
            batch_cause_word.append(batch_word_hidden[i][cs_idx,:,:])
        batch_target,batch_cause = torch.cat(batch_target,0),torch.cat(batch_cause,0)
        batch_cause_word = torch.cat(batch_cause_word,dim=0)  ## [batch_cause_num, 512, 768]
        ## cat speaker embedding
        if self.spk_embed is not None:
            tgt_spk_info = self.spk_embed(target_spk_idx)  ## [batch_size, spk_dim]
            batch_target_fl = torch.cat((tgt_spk_info,batch_target),dim=1)  ## [batch_size, 400]
            cs_spk_info = self.spk_embed(cause_spk_idx)  ## [batch_cause_num, spk_dim]
            batch_cause_fl = torch.cat((cs_spk_info,batch_cause),dim=1)  ## [batch_cause_num, 400]
        else:
            batch_target_fl = batch_target  ## [batch_size, 300]
            batch_cause_fl = batch_cause  ## [batch_cause_num, 300]
        ## cause/target utt. -> graph nodes
        target_node = self.target_proj(batch_target_fl)  ## [batch_size, 600]
        cause_node = self.cause_proj(batch_cause_fl)  ## [batch_cause_num, 600]

        # emotion trans
        emo_rep = self.emo_embed(emotion_lbs)  ## [batch_size, emo_dim]
        emotion_node = self.emo_proj(emo_rep)  ## [batch_size, 600]
        # word trans
        word_node = self.word_proj(batch_cause_word)  ## [batch_cause_num, 512, 600]
        word_mask = (label_ids != self._ignore_id).int()  ## [batch_cause_num, 512]  ## '1' indicates there is a edge between this word_node and utt node

        word_output = self.graph_network(target_node,cause_node,emotion_node,word_node,target_idx, cause_idx)[0]  ## [batch_cause_num, 512, 600]

        log_g = torch.cat((batch_cause_word,word_output),dim=-1)  ## [batch_cause_num, 512, 1200]
        log_g = self.dp_proj(log_g)

        out = self.classifier(log_g)  ## [batch_cause_num, max_seq_length, 3]
        output = out.view(-1,3)  ## [batch_cause_num * max_seq_length, 3]

        labels = label_ids.view(-1)  ## [max_seq_length * batch_cause_num]
        loss = self.criterion(output,labels)

        ## to facilitate the QA metric
        batch_dialog_turn = torch.split(input_ids,turn_num,dim=0)
        all_cause_utt = []
        for diallog,cs_id in zip(batch_dialog_turn,cause_idx):
            cs_utt = diallog[cs_id,:]
            all_cause_utt.append(cs_utt)
        ori_cause_input = torch.cat(all_cause_utt,dim=0)  ## [batch_cause_num, max_seq_len]

        ## to facilitate the case study
        new_ins_id_lis = []
        for ins_id in ins_id_lis:
            new_ins_id = split_ins_id(ins_id)
            new_ins_id_lis.extend(new_ins_id)
        assert len(new_ins_id_lis) == len(ori_cause_input)  ## batch_cause_num

        return loss, output, labels, max_step_num,ori_cause_input,new_ins_id_lis

    def weight_init(self):
        # initialize all linear layers
        init_list = [self.word_proj, self.emo_proj,
                    self.target_proj, self.cause_proj,
                    self.utt_proj,self.classifier]
        for component in init_list:
            xavier_norm(component)
        
    def feature2input(self,batch_features: List[InputFeatures],batch_cut:bool=True):
        """
        Converts a batch of InputFeatures to Tensors that can be input directly.
        """
        pad_token = 0
        all_label_ids, target_cause_id, all_context_input_ids, all_context_attention_mask, all_context_token_type_ids = [], [], [], [], []
        target_idx, cause_idx = [],[]
        all_emotion_labels = []
        tgt_spk_idx,cs_spk_idx = [],[]
        turn_num = []
        ins_id_lis = []
        for f in batch_features:
            target_id,cause_id = f.target_cause_id[0],f.target_cause_id[1:]
            target_idx.append(target_id)
            cause_idx.append(cause_id)
            tgt_spk_idx.append(f.spearker_ids[target_id])
            cs_spk_idx.extend([f.spearker_ids[t] for t in cause_id])
            all_emotion_labels.append(f.emotion_lb)
            all_label_ids.extend(f.label_ids)
            target_cause_id.append(f.target_cause_id)
            all_context_input_ids.extend(f.context_input_ids)
            all_context_attention_mask.extend(f.context_attention_mask)
            all_context_token_type_ids.extend(f.context_token_type_ids)
            turn_num.append(len(f.context_input_ids))
            ins_id_lis.append(f.ins_id)

        ## cut
        if batch_cut:
            max_real_len = bert_find_max_len(all_context_attention_mask)
            all_context_input_ids = np.array(all_context_input_ids)[:,:max_real_len]
            all_label_ids = np.array(all_label_ids)[:,:max_real_len]
            all_context_attention_mask = np.array(all_context_attention_mask)[:,:max_real_len]
            all_context_token_type_ids = np.array(all_context_token_type_ids)[:,:max_real_len]

        emotion_lbs = torch.tensor(
            all_emotion_labels,dtype=torch.int).to(self.device)
        label_ids = torch.tensor(
            all_label_ids, dtype=torch.long).to(self.device)
        context_input_ids = torch.tensor(
            all_context_input_ids, dtype=torch.long).to(self.device)
        context_attention_mask = torch.tensor(
            all_context_attention_mask, dtype=torch.float).to(self.device)
        context_token_type_ids = torch.tensor(
            all_context_token_type_ids, dtype=torch.long).to(self.device)
        target_spk_idx = torch.tensor(
            tgt_spk_idx, dtype=torch.long).to(self.device)
        cause_spk_idx = torch.tensor(
            cs_spk_idx, dtype=torch.long).to(self.device)

        assert context_token_type_ids.shape == context_input_ids.shape == context_attention_mask.shape  ## [turns_num, max_seq_length]
        
        return (context_input_ids, context_attention_mask, context_token_type_ids, turn_num,target_idx, cause_idx,target_spk_idx,cause_spk_idx,emotion_lbs,label_ids,ins_id_lis)


if __name__ == "__main__":
    ## test
    token_enocder_path = "./pretrained/language_model/spanbert-base-cased"
    embed_path = "./pretrained/word_embedding/glove.6B.300d.txt"
    emotion_lis = ["happy","surprise","disgust","anger","sad","fear"]
    emotion_dic = dict([(idx,emo) for idx,emo in enumerate(emotion_lis)])
    test_model = CausalGCN(token_enocder_path=token_enocder_path,embed_path=embed_path,emotion_dic=emotion_dic)
    # test_model.to(test_model.device)
    input_ids = torch.randint(1,20,(15,512)).long()
    attention_mask = torch.ones((15,512)).long()
    token_type_ids = torch.ones((15,512)).long()
    label_ids = torch.cat((torch.ones((5,2)),torch.ones((5,2)) * -100,torch.ones((5,12)),torch.ones((5,496)) * -100),dim=1).long()
    emotion_lbs = torch.tensor([0,3,2],dtype=torch.int)
    turn_num = [3,5,7]
    target_idx = [2,4,2]
    cause_idx = [[1],[1,2],[0,2]]
    target_spk_idx = torch.tensor([0,0,1]).long()
    cause_spk_idx = torch.tensor([1,1,0,0,0]).long()
    test_model(input_ids, attention_mask, token_type_ids, turn_num,target_idx, cause_idx,target_spk_idx,cause_spk_idx,emotion_lbs,label_ids)
    print("")
