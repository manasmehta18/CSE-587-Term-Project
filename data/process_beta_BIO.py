#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Reza

import os
import json
import re
import copy

def clean_tokenize(data, lower=False):
    ''' used to clean token, split all token with space and lower all tokens
    this function usually use in some language models which don't require strict pre-tokenization
    such as LSTM(with glove vector) or ELMO(already has tokenizer)
    :param data: string
    :return: list, contain all cleaned tokens from original input
    '''
    # split all tokens with a space
    data = re.sub(r"\.", " . ", data)
    data = re.sub(r",", " , ", data)
    data = re.sub(r"!", " ! ", data)
    data = re.sub(r"\(", " ( ", data)
    data = re.sub(r"\)", " ) ", data)
    data = re.sub(r"\?", " ? ", data)
    data = re.sub(r"\s{2,}", " ", data)
    data = data.lower() if lower else data

    # del all redundant space, split all tokens, form a list
    return [x.strip() for x in re.split('(\W+)', data) if x is not None and x.strip()]


def find_sublist(lis: list, sublis: list, i=0, j=0):
    '''
    recursion, find the **end** index of the sub-list in the list
    if multiple sub-list, calculate the first one
    if not the sub-list, there will be a IndexError
    '''
    if lis[i] == sublis[j]:
        if j == len(sublis) - 1:
            return i
        else:
            return find_sublist(lis, sublis, i + 1, j + 1)
    else:
        return find_sublist(lis, sublis, i + 1, 0)


def iob_format(causal_span: list, causal_index: list, analysis_turns: list):
    # assume that the order is in line with each other
    his_index = []
    # for those causal utterances
    for i, (index, span) in enumerate(zip(causal_index, causal_span)):
        if index == "b":
            continue
        assert span != "b"
        index = index - 1
        try:
            assert index < len(analysis_turns)
        except:
            print("there is a future causal span!")  # 2 instances
            print(analysis_turns)
            continue
        assert index + 1 == analysis_turns[index]["turn"]

        causal_str = analysis_turns[index]["utterance"]
        assert span in causal_str

        span_tk, causal_tk = span.split(), causal_str.split()
        # === del the ',' or '.' ===
        if span_tk[-1] == "," or span_tk[-1] == ".":
            span_tk = span_tk[:-1]
        if span_tk[0] == "," or span_tk[0] == ".":
            span_tk = span_tk[1:]
        # === del the ',' or '.' ===

        # find end position
        try:
            end = find_sublist(causal_tk, span_tk)
        except:
            print("warning, there may be some exceptions due to the punctuation, in IEMOCAP, so clean the tokens")
            span_tk, causal_tk = clean_tokenize(span), clean_tokenize(causal_str)
            end = find_sublist(causal_tk, span_tk)
        start = end + 1 - len(span_tk)

        # a causal utterance may have multiple causal span
        if index in his_index:
            replace_span = ['B-cause'] + ['I-cause'] * (end - start)
            assert len(analysis_turns[index]["label"][start:end + 1]) == len(replace_span)
            analysis_turns[index]["label"][start:end + 1] = replace_span
        else:
            iob_lb = []
            iob_lb += ['O'] * start
            iob_lb += ['B-cause']
            iob_lb += ['I-cause'] * (end - start)
            iob_lb += ['O'] * (len(causal_tk) - 1 - end)

            assert len(iob_lb) == len(causal_tk)

            analysis_turns[index]["token"] = causal_tk
            analysis_turns[index]["label"] = iob_lb

        his_index.append(index)

    # for those non causal utterances
    for i, turn in enumerate(analysis_turns):
        if i not in his_index:
            analysis_turns[i]["token"] = analysis_turns[i]["utterance"].split()
            analysis_turns[i]["label"] = len(analysis_turns[i]["token"]) * ['O']

    return analysis_turns


def process(data_dict: dict, path: str) -> int:
    punc = True if "iemocap" in path else False
    all_instances = []
    for _, (id, conv) in enumerate(data_dic.items()):
        all_turns = conv[0]
        his_turns = []
        for _, turn in enumerate(all_turns):
            ## === check ===
            labels.add(turn["emotion"])
            ## === check ===
            # process the punctuation in IEMOCAP
            if punc:
                turn["utterance"] = " ".join(clean_tokenize(turn["utterance"]))
                if turn.get("expanded emotion cause span", None) is not None:
                    turn["expanded emotion cause span"] = [" ".join(clean_tokenize(span)) for span in
                                                           turn["expanded emotion cause span"]]
            if turn["emotion"] == "neutral":
                his_turns.append(turn)
            else:
                try:
                    unq_cause_ut = set(turn['expanded emotion cause evidence'])
                except:
                    # print("skip, some instances in IEMOCAP don't have emotion cause annotation")
                    his_turns.append(turn)
                    continue
                if len(unq_cause_ut) == 1 and list(unq_cause_ut)[0] == "b":
                    # discard all latent
                    his_turns.append(turn)
                else:
                    causal_index = turn["expanded emotion cause evidence"]
                    causal_span = turn["expanded emotion cause span"]
                    assert len(causal_index) == len(causal_span)
                    analysis_turns = his_turns + [turn]

                    processed_turns = iob_format(causal_span, causal_index, analysis_turns)
                    processed_turns_save = copy.deepcopy(processed_turns)  # avoid any cache problem

                    instance = dict()
                    instance["id"] = path.rsplit("/")[-1].split(".")[0].split("_")[0] + "_" + id + "_utt_" + str(
                        turn["turn"])
                    instance["context"] = processed_turns_save

                    his_turns.append(turn)
                    all_instances.append(instance)

    with open(path, "w", encoding="utf-8") as w:
        json.dump(all_instances, w)

    return len(all_instances)

os.makedirs("./RECCON/beta_BIO", exist_ok=True)

data_ori = ["dailydialog_train.json", "dailydialog_valid.json", "dailydialog_test.json", "iemocap_test.json"]  
labels = set()
for part in data_ori:
    with open("./RECCON/{}".format(part), "r", encoding="utf-8") as f:
        data_dic = json.load(f)
        path = "./RECCON/beta_BIO/{}".format(part)
        ins_num = process(data_dic, path)
        print("{} process end, totally {} instances".format(part, ins_num))
print("labels:",labels)
