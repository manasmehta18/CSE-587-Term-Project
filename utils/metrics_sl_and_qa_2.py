#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this script is used for converting and calculating two sorts of metrics (sequence labeling & extractive QA).
# QA -> SL: 
## we convert the answer (bag of tokens) to IOB format. e.g., ['I','love','you'] -> [B-cause,I-cause,I-cause].
# SL -> QA: 
## because there can be multiple predicted spans with the sequence labeling, 
## in order to get as close to the experimental setup of Poria et al. as possible (only one predicted span for each question),
## we choose one span which has maximum average confidence from original model predictions to calculate QA metrics.

from posixpath import join
import torch
import numpy as np
import warnings

from torch import tensor 
from transformers import PreTrainedTokenizer

from utils.metrics_sl_and_qa import find_most_confident
from .qa_metrics import evaluate_results
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import sklearn.metrics as sk_m

def find_series(all_idx:list):
    all_span_idx = []
    span_idx = []
    for idx in all_idx:
        if len(span_idx) == 0 or span_idx[-1] + 1 == idx:
            span_idx.append(idx)
        else:
            all_span_idx.append(span_idx)
            span_idx = [idx]
    if len(span_idx) > 0:
        all_span_idx.append(span_idx)
    return all_span_idx

def find_longest_match(span_idx:list,st_lis:list,ed_lis:list):
    best_score = -9999
    best_st,best_ed = None,None
    for st,ed in zip(st_lis,ed_lis):
        t = list(range(st,ed))
        assert st <= ed
        overlap=[idx for idx in span_idx if idx in t]
        score = len(overlap)
        if score > best_score:
            best_score = score
            best_st,best_ed = st,ed

    return best_st,best_ed

def generate_all_spans(ins_tk_ids:torch.tensor,tokenizer:PreTrainedTokenizer,ins_iob:list):
    all_spans = []
    span = []
    ins_tk_ids = ins_tk_ids.numpy().tolist()
    for tk_dis,iob in zip(ins_tk_ids,ins_iob):
        if iob in ["I-cause","B-cause"]:
            span.append(tokenizer.convert_ids_to_tokens(tk_dis))
        else:
            if len(span) != 0:
                all_spans.append(" ".join(span))
                span = []
            else:
                continue
    if len(span) != 0:
        all_spans.append(" ".join(span))

    return all_spans

def span_metric(input_ids:tensor,output:tensor,label:tensor,seq_steps:list,labels_lis:list,max_seq_length:int,tokenizer:PreTrainedTokenizer,ins_ids:list,return_case:bool=False):
    '''
    calculate EM, SQuAD F1, imp F1, overall F1 and span F1
    len(label) == len(pred) == instance_num * max_seq_length
    '''
    convert_ids_to_tokens = tokenizer.convert_ids_to_tokens
    label_map = {i:lb for i,lb in enumerate(labels_lis)}
    label_map_rev = {lb:i for i,lb in enumerate(labels_lis)}
    O_ids = label_map_rev["O"]  ## ids corresponding to 'O'
    # assert label.shape[0] % max_seq_length == 0
    assert label.shape[0] == input_ids.shape[0] == output.shape[0]
    all_ins_num = len(seq_steps)

    confident, prediction = torch.max(output,dim=1)

    correct,incorrect,similar = 0,0,0
    correct_text,incorrect_text,similar_text = {},{},{}
    correct_case,incorrect_case,similar_case = {},{},{}
    all_pred_iob = []
    all_true_iob = [] 
    all_pred_ids = []
    all_true_ids = []
    current_step = 0
    pred_span_num, ground_span_num = 0,0
    assert len(seq_steps) == len(ins_ids)  ## instance num
    # for each instance
    for i,step in enumerate(seq_steps):
        ins_id = ins_ids[i]
        ins_tk_ids = input_ids[current_step:current_step+step]
        ins_label = label[current_step:current_step+step]
        ins_pred = prediction[current_step:current_step+step]
        ins_conf = confident[current_step:current_step+step]
        current_step += step
        # del all useless tokens (label is -100)
        selec_index=torch.tensor(np.squeeze(np.argwhere(np.array(ins_label)!=-100)),dtype=torch.int)
        tmp_ins_tk_ids = torch.index_select(ins_tk_ids,0,selec_index)
        tmp_ins_label = torch.index_select(ins_label,0,selec_index)
        tmp_ins_pred = torch.index_select(ins_pred,0,selec_index)
        tmp_ins_conf = torch.index_select(ins_conf,0,selec_index)
        # calculate label IOB
        ins_true_iob = [label_map[lb_ids] for lb_ids in tmp_ins_label.tolist()]
        ins_pred_iob = [label_map[lb_ids] for lb_ids in tmp_ins_pred.tolist()]
        assert len(ins_true_iob) == len(ins_pred_iob)
        all_true_iob.append(ins_true_iob)
        all_pred_iob.append(ins_pred_iob)
        # calculate label ids
        all_true_ids.extend([lb_ids for lb_ids in tmp_ins_label.tolist()])
        all_pred_ids.extend([lb_ids for lb_ids in tmp_ins_pred.tolist()])
        # select prediction span
        all_pred_idx=np.squeeze(np.argwhere(np.array(tmp_ins_pred)!=O_ids)).tolist()
        all_pred_idx=[all_pred_idx] if type(all_pred_idx) == int else all_pred_idx 
        st_lis,ed_lis = [],[]
        if len(all_pred_idx) != 0:
            all_span_idx = find_series(all_pred_idx)
            pred_span_num += len(all_span_idx)
            # best_st ,best_ed,best_span_conf = -1,-1,-1
            for each_span in all_span_idx:
                st = each_span[0]
                ed = st + len(each_span)
                st_lis.append(st)
                ed_lis.append(ed)
        else:
            pred_tk, pred_span = [], ""
        # search groud truth span
        all_answer_idx=np.squeeze(np.argwhere(np.array(tmp_ins_label)!=O_ids)).tolist()
        all_answer_idx = [all_answer_idx] if type(all_answer_idx) == int else all_answer_idx 
        answer_span_lis,answer_tk_lis = [],[]
        pred_span_lis,pred_tk_lis = [],[]
        id_lis = []
        # select prediction
        if len(all_answer_idx) != 0:
            ## process pos samples
            all_ans_span_idx=find_series(all_answer_idx)
            ground_span_num += len(all_ans_span_idx)
            for j,span_idx in enumerate(all_ans_span_idx):
                id = "true_"+ str(i) +"_span"+"_"+str(j)
                id_lis.append(id)
                ### select the span which has the maximum prediction confidence
                best_st,best_ed = find_most_confident(tmp_ins_conf.tolist(),st_lis,ed_lis)
                if best_st is not None and best_ed is not None:
                    pred_tk_ids=tmp_ins_tk_ids.tolist()[best_st:best_ed]
                    pred_tk = convert_ids_to_tokens(pred_tk_ids)
                    pred_tk_lis.append(pred_tk)
                    pred_span = " ".join(pred_tk)
                    pred_span_lis.append(pred_span)  
                else:
                    pred_tk_lis.append([])
                    pred_span_lis.append("")
                answer_tk_ids=tmp_ins_tk_ids.tolist()[span_idx[0]:span_idx[-1]+1]
                answer_tk = convert_ids_to_tokens(answer_tk_ids)
                answer_tk_lis.append(answer_tk)
                answer_span = " ".join(answer_tk)
                answer_span_lis.append(answer_span)
        else:
            ## process neg samples, though we don't care
            id = "impossible"+"_"+str(i)
            id_lis.append(id)
            answer_tk, answer_span = [], ""
            answer_tk_lis.append(answer_tk)
            answer_span_lis.append(answer_span)
            if len(st_lis) !=0 and len(ed_lis) != 0:
                pred_tk_ids=tmp_ins_tk_ids.tolist()[st_lis[0]:ed_lis[0]]
                pred_tk = convert_ids_to_tokens(pred_tk_ids)
                pred_tk_lis.append(pred_tk)
                pred_span = " ".join(pred_tk)
                pred_span_lis.append(pred_span) 
            else:
                pred_tk, pred_span = [], ""
                pred_tk_lis.append(pred_tk)
                pred_span_lis.append(pred_span)
        assert len(id_lis) == len(answer_span_lis) == len(pred_span_lis)
        # generate all case for analysis
        all_answer_spans = generate_all_spans(tmp_ins_tk_ids,tokenizer,ins_true_iob)
        all_pred_spans = generate_all_spans(tmp_ins_tk_ids,tokenizer,ins_pred_iob)  
        ins_target_id,ins_cause_id = ins_id.split("_cause_utt_")[0],ins_id.split("_cause_utt_")[1]
        ins_cause_id = "cause_utt_" + ins_cause_id
        # divide correct, similar and incorrect instance, similar to Poria et al. (https://github.com/declare-lab/RECCON/)
        for p in range(len(id_lis)):
            id = id_lis[p]
            pred_span = pred_span_lis[p]
            answer_span = answer_span_lis[p]
            if pred_span == answer_span:
                correct += 1
                ## correct_text[id] = answer_span
                correct_text[id] = {
                    "truth": answer_span,
                    "predicted": pred_span
                }
                ## append case
                if correct_case.get(ins_target_id,None) is None:
                    correct_case[ins_target_id] = {ins_cause_id:{"truth": all_answer_spans,"predicted": all_pred_spans}}
                else:
                    correct_case[ins_target_id][ins_cause_id] = {"truth": all_answer_spans,"predicted": all_pred_spans}
            elif pred_span in answer_span or answer_span in pred_span:
                similar += 1
                similar_text[id] = {
                    "truth": answer_span,
                    "predicted": pred_span
                }
                ## append case
                if similar_case.get(ins_target_id,None) is None:
                    similar_case[ins_target_id] = {ins_cause_id:{"truth": all_answer_spans,"predicted": all_pred_spans}}
                else:
                    similar_case[ins_target_id][ins_cause_id] = {"truth": all_answer_spans,"predicted": all_pred_spans}
            else:
                incorrect += 1
                incorrect_text[id] = {
                    "truth": answer_span,
                    "predicted": pred_span
                }
                ## append case
                if incorrect_case.get(ins_target_id,None) is None:
                    incorrect_case[ins_target_id] = {ins_cause_id:{"truth": all_answer_spans,"predicted": all_pred_spans}}
                else:
                    incorrect_case[ins_target_id][ins_cause_id] = {"truth": all_answer_spans,"predicted": all_pred_spans}
    assert current_step == input_ids.shape[0]
    # calculate span F1
    span_f1 = f1_score(all_true_iob,all_pred_iob)
    span_prec = precision_score(all_true_iob,all_pred_iob)
    span_rec = recall_score(all_true_iob,all_pred_iob)
    span_detection = {"Span F1":span_f1*100,
                    "Span Precision":span_prec*100,
                    "Span Recall":span_rec*100}
    # optional: calculate token F1
    token_cls = {"Token Micro F1": sk_m.f1_score(all_true_ids, all_pred_ids, average="micro")*100,
                "Token Macro F1": sk_m.f1_score(all_true_ids, all_pred_ids, average="macro")*100,
                "Token Accuracy":accuracy_score(all_true_iob,all_pred_iob)*100}
    # calculate EM, SQuAD F1, imp F1 and overall F1
    QA_formulation = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }
    _,extractive_qa = evaluate_results(QA_formulation)
    
    result = dict(span_detection,**extractive_qa)
    result = dict(result,**token_cls)

    # print("==> total prediction span num ",pred_span_num)
    # print("==> total ground truth span num ",ground_span_num)

    if not return_case:
        return result
    else:
        all_cases = {
            "correct_text": correct_case,
            "similar_text": similar_case,
            "incorrect_text": incorrect_case,
        }
        return result,all_cases


def span_metric_for_embed(input_ids:tensor,output:tensor,label:tensor,seq_steps:list,labels_lis:list,max_seq_length:int,word2dix:dict):
    '''
    calculate EM, SQuAD F1, imp F1, overall F1 and span F1
    len(label) == len(pred) == instance_num * max_seq_length
    this function play a role, when the token encoder is GRU or LSTM 
    '''
    # convert_ids_to_tokens = tokenizer.convert_ids_to_tokens
    idx2word = {v:k for k,v in word2dix.items()}
    idx2word[1] = "unk"
    convert_ids_to_tokens = lambda x:[idx2word[t] for t in x]
    label_map = {i:lb for i,lb in enumerate(labels_lis)}
    label_map_rev = {lb:i for i,lb in enumerate(labels_lis)}
    O_ids = label_map_rev["O"]  ## ids corresponding to 'O'
    # assert label.shape[0] % max_seq_length == 0
    assert label.shape[0] == input_ids.shape[0] == output.shape[0]
    all_ins_num = len(seq_steps)

    confident, prediction = torch.max(output,dim=1)

    correct,incorrect,similar = 0,0,0
    correct_text,incorrect_text,similar_text = {},{},{}
    all_pred_iob = []
    all_true_iob = [] 
    all_pred_ids = []
    all_true_ids = []
    # for each instance
    current_step = 0
    for i,step in enumerate(seq_steps):
        ins_tk_ids = input_ids[current_step:current_step+step]
        ins_label = label[current_step:current_step+step]
        ins_pred = prediction[current_step:current_step+step]
        ins_conf = confident[current_step:current_step+step]
        current_step += step
        # del all useless tokens
        selec_index=torch.tensor(np.squeeze(np.argwhere(np.array(ins_label)!=-100)),dtype=torch.int)
        tmp_ins_tk_ids = torch.index_select(ins_tk_ids,0,selec_index)
        tmp_ins_label = torch.index_select(ins_label,0,selec_index)
        tmp_ins_pred = torch.index_select(ins_pred,0,selec_index)
        tmp_ins_conf = torch.index_select(ins_conf,0,selec_index)
        # calculate label IOB
        ins_true_iob = [label_map[lb_ids] for lb_ids in tmp_ins_label.tolist()]
        ins_pred_iob = [label_map[lb_ids] for lb_ids in tmp_ins_pred.tolist()]
        assert len(ins_true_iob) == len(ins_pred_iob)
        all_true_iob.append(ins_true_iob)
        all_pred_iob.append(ins_pred_iob)
        # calculate label ids
        all_true_ids.extend([lb_ids for lb_ids in tmp_ins_label.tolist()])
        all_pred_ids.extend([lb_ids for lb_ids in tmp_ins_pred.tolist()])
        # select prediction span
        all_pred_idx=np.squeeze(np.argwhere(np.array(tmp_ins_pred)!=O_ids)).tolist()
        all_pred_idx=[all_pred_idx] if type(all_pred_idx) == int else all_pred_idx 
        st_lis,ed_lis = [],[]
        if len(all_pred_idx) != 0:
            all_span_idx = find_series(all_pred_idx)
            # conf = tmp_ins_conf.tolist()
            # best_st ,best_ed,best_span_conf = -1,-1,-1
            for each_span in all_span_idx:
                st = each_span[0]
                ed = st + len(each_span)
                st_lis.append(st)
                ed_lis.append(ed)
        else:
            pred_tk, pred_span = [], ""
        # search groud truth span
        all_answer_idx=np.squeeze(np.argwhere(np.array(tmp_ins_label)!=O_ids)).tolist()
        all_answer_idx = [all_answer_idx] if type(all_answer_idx) == int else all_answer_idx 
        answer_span_lis,answer_tk_lis = [],[]
        pred_span_lis,pred_tk_lis = [],[]
        id_lis = []
        # select prediction
        if len(all_answer_idx) != 0:
            ## process pos samples
            all_ans_span_idx=find_series(all_answer_idx)
            for j,span_idx in enumerate(all_ans_span_idx):
                id = "true_"+ str(i) +"_span"+"_"+str(j)
                id_lis.append(id)
                ### select the span which has the maximum prediction confidence
                best_st,best_ed = find_most_confident(tmp_ins_conf.tolist(),st_lis,ed_lis)
                if best_st is not None and best_ed is not None:
                    pred_tk_ids=tmp_ins_tk_ids.tolist()[best_st:best_ed]
                    pred_tk = convert_ids_to_tokens(pred_tk_ids)
                    pred_tk_lis.append(pred_tk)
                    pred_span = " ".join(pred_tk)
                    pred_span_lis.append(pred_span)  
                else:
                    pred_tk_lis.append([])
                    pred_span_lis.append("")
                answer_tk_ids=tmp_ins_tk_ids.tolist()[span_idx[0]:span_idx[-1]+1]
                answer_tk = convert_ids_to_tokens(answer_tk_ids)
                answer_tk_lis.append(answer_tk)
                answer_span = " ".join(answer_tk)
                answer_span_lis.append(answer_span)
        else:
            ## process neg samples, though we don't care
            id = "impossible"+"_"+str(i)
            id_lis.append(id)
            answer_tk, answer_span = [], ""
            answer_tk_lis.append(answer_tk)
            answer_span_lis.append(answer_span)
            if len(st_lis) !=0 and len(ed_lis) != 0:
                pred_tk_ids=tmp_ins_tk_ids.tolist()[st_lis[0]:ed_lis[0]]
                pred_tk = convert_ids_to_tokens(pred_tk_ids)
                # pred_tk = idx2word[pred_tk_ids]
                pred_tk_lis.append(pred_tk)
                pred_span = " ".join(pred_tk)
                pred_span_lis.append(pred_span) 
            else:
                pred_tk, pred_span = [], ""
                pred_tk_lis.append(pred_tk)
                pred_span_lis.append(pred_span)
        assert len(id_lis) == len(answer_span_lis) == len(pred_span_lis)
        # divide correct, similar and incorrect instance
        for p in range(len(id_lis)):
            id = id_lis[p]
            pred_span = pred_span_lis[p]
            answer_span = answer_span_lis[p]
            if pred_span == answer_span:
                correct += 1
                # correct_text[id] = answer_span
                correct_text[id] = {
                    "truth": answer_span,
                    "predicted": pred_span
                }
            elif pred_span in answer_span or answer_span in pred_span:
                similar += 1
                similar_text[id] = {
                    "truth": answer_span,
                    "predicted": pred_span
                }
            else:
                incorrect += 1
                incorrect_text[id] = {
                    "truth": answer_span,
                    "predicted": pred_span
                }
    assert current_step == input_ids.shape[0]
    # calculate span F1
    span_f1 = f1_score(all_true_iob,all_pred_iob)
    span_prec = precision_score(all_true_iob,all_pred_iob)
    span_rec = recall_score(all_true_iob,all_pred_iob)
    span_detection = {"Span F1":span_f1*100,
                    "Span Precision":span_prec*100,
                    "Span Recall":span_rec*100}
    # optional: calculate token F1
    token_cls = {"Token Micro F1": sk_m.f1_score(all_true_ids, all_pred_ids, average="micro")*100,
                "Token Macro F1": sk_m.f1_score(all_true_ids, all_pred_ids, average="macro")*100,
                "Token Accuracy":accuracy_score(all_true_iob,all_pred_iob)*100}
    # calculate EM, SQuAD F1, imp F1 and overall F1
    QA_formulation = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }
    _,extractive_qa = evaluate_results(QA_formulation)
    
    result = dict(span_detection,**extractive_qa)
    result = dict(result,**token_cls)

    return result
    