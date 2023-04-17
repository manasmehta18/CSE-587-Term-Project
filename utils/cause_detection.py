# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## change the code for the causal span detection task

import logging
import os
import json
import copy
import torch
import numpy as np

from re import T
from torch import nn
from torch.utils.data import Dataset, DataLoader,SequentialSampler
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.optimization import AdamW
from dataclasses import dataclass
from typing import List, Optional, Union
from filelock import FileLock

logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    context_input_ids: Optional[List[List[int]]] = None
    context_attention_mask: Optional[List[List[int]]] = None
    context_token_type_ids: Optional[List[List[int]]] = None
    target_cause_id: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    speaker_context_input_dis : Optional[List[List[int]]] = None
    spearker_ids : Optional[List[int]] = None
    current_speaker : Optional[int] = None
    target_id: Optional[int] = None
    emotion_lb : Optional[int] = None
    true_token_num : Optional[List[int]] = None
    position_ids : Optional[List[int]] = None
    ins_id : Optional[str] = None

class CauseDetectionDataset_GCN(Dataset):
    """
    dataset class, each instance in this class is the whole conversation context (i.e., all utternaces in the dialogue)
    this class is helpful due to we have the hierarchy encoder which need encode the whole dialogue 
    """
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_path: str,
        context_path: str,
        tokenizer_path: str,
        labels: List[str],
        model_type: str,
        max_seq_length: int = 512,
        emotion_labels: List[str] = None,
        overwrite_cache=False,
        shuffle=False
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Load data features from cache or dataset file
        data_dir = data_path.rsplit("/", 1)[0]
        data_name = data_path.rsplit("/", 1)[-1].split(".")[0]
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                data_name, tokenizer.__class__.__name__, str(max_seq_length)),
        )
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(
                    f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(
                    f"Creating features from dataset file at {data_dir}")
                self.features = convert_raw_data_to_features_gcn(
                    data_path,
                    context_path,
                    labels,
                    emotion_labels,
                    max_seq_length,
                    tokenizer,
                    cls_token_at_end=bool(model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(
                    f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)
        if shuffle:
            np.random.shuffle(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

def convert_raw_data_to_features_gcn(
    data_path: str,
    context_path: str,
    label_list: List[str],
    emotion_labels: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:

    label_map = {label: i for i, label in enumerate(label_list)}
    speaker_map = {"A":0,"B":1}
    emotion_labels = ["happy","surprise","disgust","anger","sad","fear"] if emotion_labels is None else emotion_labels  ## use labels from DD by default
    emotion_map = {emotion:i for i,emotion in enumerate(emotion_labels)}

    # read and process the data
    with open(data_path, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
    with open(context_path, "r", encoding="utf-8") as context_file:
        context = json.load(context_file)

    cnt = 0
    features = []
    for _, dialog in enumerate(context):
        # process context
        context_ids = []
        speaker_ids = []
        context_attention = []
        context_token_type = []
        for _, item in enumerate(dialog["context"]):
            h_tk = tokenizer.tokenize(item["utterance"])
            rs_tk, rs_seg = [], []
            for tk in h_tk:
                tks = tokenizer.tokenize(tk)
                if len(tks) > 0:
                    rs_tk.extend(tks)
                    rs_seg.extend([sequence_a_segment_id]*len(tks))
            special_cnt = tokenizer.num_special_tokens_to_add()
            assert len(rs_tk) == len(rs_seg)
            if len(rs_tk) > max_seq_length - special_cnt:
                rs_tk = rs_tk[: (max_seq_length - special_cnt)]
                rs_seg = rs_seg[: (max_seq_length - special_cnt)]
            rs_tk = [cls_token] + rs_tk
            rs_seg = [sequence_a_segment_id] + rs_seg
            rs_tk += [sep_token]
            rs_seg += [sequence_a_segment_id]
            rs_ids = tokenizer.convert_tokens_to_ids(rs_tk)
            rs_mask = [1 if mask_padding_with_zero else 0] * len(rs_ids)
            padding_length = max_seq_length - len(rs_ids)
            rs_ids += [pad_token] * padding_length
            rs_mask += [0 if mask_padding_with_zero else 1] * padding_length
            rs_seg += [pad_token_segment_id] * padding_length

            context_ids.append(rs_ids)
            context_attention.append(rs_mask)
            context_token_type.append(rs_seg)

            speaker_ids.append(speaker_map[item["speaker"]])
        # process cause, there are multiple cause utt. in each instance
        for _, ins_id in enumerate(dialog["contain_list"]):
            if cnt % 1000 == 0:
                logger.info("Writing example %d of %d", cnt, len(data))
            cnt += 1
            data_ins = data[ins_id]
            emotion = data_ins["emotion"]
            if emotion in emotion_labels:
                emotion_lb = emotion_map[emotion]
            elif emotion in ["excited","frustrated"]:
                sim_lb_trans = {"excited":"happy","frustrated":"sad"}  ## deal with the RECCON-IE
                emotion = sim_lb_trans[emotion]
                emotion_lb = emotion_map[emotion]
            else:
                raise RuntimeError("there is no such emotion `{}` in the pre-defined categories".format(emotion))
                
            cause = data_ins["cause"]
            target = data_ins["target"]
            labels = data_ins["label"]
            target_id = data_ins["target_id"] - 1
            current_speaker = speaker_map[dialog["context"][target_id]["speaker"]]
            assert target_id < len(context_ids)
            cause_id = [t-1 for t in data_ins["cause_id"]]
            for cs_id in cause_id:
                assert cs_id < len(context_ids)
            target_cause_id = [target_id] + cause_id
            assert len(cause_id) == len(cause)
            try:
                assert len(cause) > 0
            except:
                print("skip instance:",data_ins)  ## some have no cause annotations
                continue

            label_ids = []
            tokens = []
            segment_ids = []
            all_mask = []
            for cs_cnt,(cs, lb) in enumerate(zip(cause, labels)):
                label_id = []
                token_id = []
                seg_id = []
                for word,label in zip(cs,lb):
                    word_tokens = tokenizer.tokenize(word)
                    # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                    if len(word_tokens) > 0:
                        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                        lb_ids = [label_map[label]] + \
                            [pad_token_label_id] * (len(word_tokens) - 1)
                        label_id.extend(lb_ids)
                        token_id.extend(word_tokens)
                        seg_id.extend([sequence_a_segment_id]*len(lb_ids))
                special_tokens_count = tokenizer.num_special_tokens_to_add()
                if len(token_id) > max_seq_length - special_tokens_count:
                    token_id = token_id[: (max_seq_length - special_tokens_count)]
                    label_id = label_id[: (
                        max_seq_length - special_tokens_count)]
                    seg_id = seg_id[: (
                        max_seq_length - special_tokens_count)]
                token_id += [sep_token]
                label_id += [pad_token_label_id]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    token_id += [sep_token]
                    label_id += [pad_token_label_id]
                seg_id += [sequence_a_segment_id]
                assert len(seg_id) == len(token_id)

                if cls_token_at_end:
                    token_id += [cls_token]
                    label_id += [pad_token_label_id]
                    seg_id += [cls_token_segment_id]
                else:
                    token_id = [cls_token] + token_id
                    label_id = [pad_token_label_id] + label_id
                    seg_id = [cls_token_segment_id] + seg_id
                
                token_id = tokenizer.convert_tokens_to_ids(token_id)
                input_mask = [1 if mask_padding_with_zero else 0] * len(token_id)
                padding_length = max_seq_length - len(token_id)
                if pad_on_left:
                    token_id = ([pad_token] * padding_length) + token_id
                    input_mask = ([0 if mask_padding_with_zero else 1]
                                * padding_length) + input_mask
                    seg_id = ([pad_token_segment_id] *
                                padding_length) + seg_id
                    label_id = ([pad_token_label_id] * padding_length) + label_id
                else:
                    token_id += [pad_token] * padding_length
                    input_mask += [0 if mask_padding_with_zero else 1] * \
                        padding_length
                    seg_id += [pad_token_segment_id] * padding_length
                    label_id += [pad_token_label_id] * padding_length

                assert len(token_id) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(seg_id) == max_seq_length
                assert len(label_id) == max_seq_length

                tokens.append(token_id)
                segment_ids.append(seg_id)
                label_ids.append(label_id)
                all_mask.append(input_mask)

            ## replace the context
            cp_context_ids = copy.deepcopy(context_ids)
            cp_context_attention = copy.deepcopy(context_attention)
            cp_context_token_type = copy.deepcopy(context_token_type)

            assert len(tokens) == len(cause_id)  ## cause_num
            for i,id in enumerate(cause_id):
                assert len(cp_context_ids[id]) == len(tokens[i])
                assert len(cp_context_attention[id]) == len(all_mask[i])
                assert len(cp_context_token_type[id]) == len(segment_ids[i])
                cp_context_ids[id] = tokens[i]
                cp_context_attention[id] = all_mask[i]
                cp_context_token_type[id] = segment_ids[i]

            features.append(
                InputFeatures(
                    context_input_ids=cp_context_ids, context_attention_mask=cp_context_attention, context_token_type_ids=cp_context_token_type,
                    label_ids=label_ids,target_cause_id=target_cause_id,spearker_ids=speaker_ids,emotion_lb=emotion_lb,ins_id = ins_id
                )
            )

    return features