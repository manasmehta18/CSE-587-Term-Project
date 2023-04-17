#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Reza
# this script was created for tagging the IOB label

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
    # data = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", data)
    data = re.sub(r"\'s", " \'s", data)
    data = re.sub(r"n\'t", " not", data)
    data = re.sub(r"\'ve", " have", data)
    data = re.sub(r"\'re", " are", data)
    data = re.sub(r"\'d", " would", data)
    data = re.sub(r"\'ll", " will", data)
    data = re.sub(r"\'m", " am", data)
    data = re.sub(r"\.", " . ", data)
    data = re.sub(r",", " , ", data)
    data = re.sub(r"!", " ! ", data)
    data = re.sub(r"\(", " ( ", data)
    data = re.sub(r"\)", " ) ", data)
    data = re.sub(r"\?", " ? ", data)
    data = re.sub(r"\s{2,}", " ", data)
    data = data.lower() if lower else data

    # split all tokens, form a list
    return [x.strip() for x in re.split('(\W+)', data) if x.strip()]


def process_ori(data_ori: dict, all_samples: dict):
    for i, (id, all_turn) in enumerate(data_ori.items()):
        turns = all_turn[0]
        rs_ins = dict()
        rs_ins["id"] = id
        rs_ins["context"] = turns
        rs_ins["contain_list"] = []

        # process all turns to tokens for convenience
        for turn in rs_ins["context"]:
            turn["token"] = clean_tokenize(turn["utterance"])

        for _, ins_id in enumerate(all_samples.keys()):
            if id in ins_id:
                right_bound=ins_id[ins_id.index(id)+len(id)]
                left_bound=ins_id[ins_id.index(id)-1]
                if "dailydialog" in ins_id and (right_bound != "_" or left_bound != "_"):
                    # print("skip ambiguous id:")
                    # print("id:{}\tins_id:{}".format(id,ins_id))
                    continue
                rs_ins["contain_list"].append(ins_id)
        yield rs_ins


def process_pos(data_pos:list,threshold:int=1):
    for i, ins in enumerate(data_pos):
        id = ins["id"]
        context = ins["context"]
        target_turn = context[-1]
        target_em = target_turn["emotion"]
        assert target_em != "neutral"
        target_tk = []
        for tk in target_turn["token"]:
            target_tk += clean_tokenize(tk)
        all_rs_tk = []
        all_rs_lb = []
        cause_id_lis = []
        for j, turn in enumerate(context):
            unq_lb = list(set(turn["label"]))
            if len(unq_lb) == 1 and unq_lb[0] == "O":
                continue
            # if len([lb for lb in turn["label"] if lb == 'B-cause']) > 1:
            #     print("\n==>there are multi cause in one cause utt.")
            #     print(turn)
            rs_tk = []
            rs_lb = []
            assert len(turn["token"]) == len(turn["label"])
            for idx, tk in enumerate(turn["token"]):
                tks = clean_tokenize(tk)
                lbs = [turn["label"][idx]]
                if len(tks) > 1:
                    if lbs[0] in ["B-cause", "I-cause"]:
                        lbs += ["I-cause"] * (len(tks)-1)
                    else:
                        lbs += ["O"] * (len(tks)-1)
                elif len(tks) == 1:
                    lbs = lbs
                else:
                    raise RuntimeError
                rs_tk += tks
                rs_lb += lbs

            # rs_tk.append("/")
            # rs_lb.append("O")
            assert len(rs_tk) == len(rs_lb)

            all_rs_tk.append(rs_tk)
            all_rs_lb.append(rs_lb)
            cause_id_lis.append(str(j+1))

            # rs_ins = dict()
            # rs_id = id + "_cause_utt_" + str(j+1)
            # # rs_ins["id"] = id + "_cause_utt_" + str(j+1)
            # rs_ins["target"] = target_tk
            # rs_ins["target_id"] = len(context)
            # rs_ins["emotion"] = target_em
            # rs_ins["cause"] = rs_tk
            # rs_ins["cause_id"] = j+1
            # rs_ins["label"] = rs_lb
        rs_ins = dict()
        rs_id = id + "_cause_utt_" + "_".join(cause_id_lis)
        rs_ins["target"] = target_tk
        rs_ins["target_id"] = len(context)
        rs_ins["emotion"] = target_em
        rs_ins["cause"] = all_rs_tk
        rs_ins["cause_id"] = [int(idx) for idx in cause_id_lis]
        rs_ins["label"] = all_rs_lb
        
        if len(rs_ins["cause"]) >= threshold:
            yield rs_id, rs_ins
        else:
            continue

def process_imp(data_imp: list):
    for i, ins in enumerate(data_imp):
        cause = ins["context"]
        id = ins["qas"][0]["id"]
        if "impossible" not in id:
            continue
        question = ins["qas"][0]["question"].split(" ")
        target = " ".join(question[4:len(question)-17])
        emotion = question[-2]
        target_tk, causal_tk = clean_tokenize(target), clean_tokenize(cause)

        rs_ins = dict()
        rs_ins["target"] = target_tk
        rs_ins["target_id"] = int(id.split("_")[-5])
        rs_ins["emotion"] = emotion
        rs_ins["cause"] = causal_tk
        rs_ins["cause_id"] = int(id.split("_")[-1])
        rs_ins["label"] = ["O"] * len(causal_tk)

        yield id, rs_ins


os.makedirs("./RECCON_BIO/", exist_ok=True)
save_path = "./RECCON_BIO/"
# threshold = 6
# os.makedirs("./RECCON_BIO/{}".format(threshold), exist_ok=True)
# save_path = "./RECCON_BIO/{}/".format(threshold)

impossible_DD_train = "./RECCON/qa/dailydialog_qa_train_without_context.json"
impossible_DD_test = "./RECCON/qa/dailydialog_qa_test_without_context.json"
impossible_DD_eval = "./RECCON/qa/dailydialog_qa_valid_without_context.json"
impossible_IE_test = "./RECCON/qa/iemocap_qa_test_without_context.json"

ori_DD_train = "./RECCON/dailydialog_train.json"
ori_DD_eval = "./RECCON/dailydialog_valid.json"
ori_DD_test = "./RECCON/dailydialog_test.json"
ori_IE_test = "./RECCON/iemocap_test.json"

pos_DD_train = "./RECCON/beta_BIO/dailydialog_train.json"
pos_DD_eval = "./RECCON/beta_BIO/dailydialog_valid.json"
pos_DD_test = "./RECCON/beta_BIO/dailydialog_test.json"
pos_IE_test = "./RECCON/beta_BIO/iemocap_test.json"

data_pos_set = [pos_DD_train] + [pos_DD_eval] + [pos_DD_test] + [pos_IE_test]
data_impossible_set = [impossible_DD_train] + [impossible_DD_eval] + \
    [impossible_DD_test] + [impossible_IE_test]
data_ori_set = [ori_DD_train] + [ori_DD_eval] + \
    [ori_DD_test] + [ori_IE_test]

for i, (pos, imp) in enumerate(zip(data_pos_set, data_impossible_set)):
    ori = data_ori_set[i]
    all_ori = open(ori, "r", encoding="utf-8")
    all_pos = open(pos, "r", encoding="utf-8")
    all_imp = open(imp, "r", encoding="utf-8")

    data_ori = json.load(all_ori)
    data_pos = json.load(all_pos)
    data_imp = json.load(all_imp)

    rs_pos = {rs_id: rs_ins for rs_id, rs_ins in process_pos(data_pos)}
    # rs_imp = {rs_id: rs_ins for rs_id, rs_ins in process_imp(data_imp)}
    rs_imp = {}
    all_samples = dict(rs_pos, **rs_imp)

    # append all possible/impossible instances id to rs_whole_dialog
    rs_whole_dialog = list(process_ori(data_ori, all_samples))

    print("==> for {} data:".format(pos.rsplit("/", 1)[-1]))
    print("num of positive samples:", len(rs_pos))
    print("num of negative samples:", len(rs_imp))
    print("num of dialogue:", len(rs_whole_dialog))

    ## check the ins num ==========
    # all_pos_num = 0
    # for item in rs_whole_dialog:
    #     all_pos_num += len(item['contain_list'])
    # assert len(rs_pos) == all_pos_num
    # print("* * check pos num:",all_pos_num)
    ## ============================

    save_ins_name = ori.rsplit("/", 1)[-1]
    # save_ins_name = "new_"+ori.rsplit("/", 1)[-1]
    with open(save_path+save_ins_name, "w", encoding="utf-8") as tg_data:
        json.dump(all_samples, tg_data)
    save_dialog_name = ori.rsplit(
        "/", 1)[-1].split(".")[0]+"_shared_dialog.json"
    with open(save_path+save_dialog_name, "w", encoding="utf-8") as tg_context:
        json.dump(rs_whole_dialog, tg_context)

    all_pos.close()
    all_imp.close()
    all_ori.close()
