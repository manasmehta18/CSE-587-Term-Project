import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  ## enable the deterministic method
import numpy as np
import random
import json
import argparse
import logging
import torch
import transformers

from torch import nn
from urllib import parse
from re import L
from dataclasses import dataclass, field
from importlib import import_module, util
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm,trange, utils
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers.trainer_utils import is_main_process
from tensorboardX import SummaryWriter
from models.Dialogue_McIN import CausalGCN
from utils.tool_box import count_parameters,show_parameters,seed_torch
from utils.cause_detection import CauseDetectionDataset_GCN
from utils.metrics_sl_and_qa_2 import span_metric, span_metric_for_embed

logger = logging.getLogger(__name__)


def test_model(model,args,test_data,labels_lis,data_name):
    # evaluate model on test set
    ts_loss = 0.
    test_input, test_output, test_label = [],[],[]
    seq_steps = []
    ins_ids = []
    model.eval()
    with torch.no_grad():
        for step_test,start_index_test in enumerate(tqdm(range(0,len(test_data),args.eval_batch))):
            batch_features=test_data[start_index_test: min(start_index_test + args.eval_batch, len(test_data))]
            if len(batch_features) == 0:
                continue
            batch_input=model.feature2input(batch_features)
            loss,output,labels,max_step_num,ori_cause_input,batch_ins_ids = model(*batch_input)

            ts_loss += loss.item()
            seq_steps.extend([max_step_num]*len(ori_cause_input))

            output = output.detach().to("cpu")
            labels = labels.detach().to("cpu")
            input_ids = ori_cause_input.detach().to("cpu")[:,0:max_step_num].contiguous().view(-1)

            test_output.append(output)
            test_label.append(labels)
            test_input.append(input_ids)
            assert output.shape[0] == labels.shape[0] == input_ids.shape[0]
            
            ins_ids.extend(batch_ins_ids)
    logger.info("avg test loss:{:f}".format(ts_loss/(step_test+1)))
    # cauculate metric
    tmp_input_ids = torch.cat(test_input,dim=0)
    tmp_label = torch.cat(test_label,dim=0)
    temp_output = torch.cat(test_output,dim=0)
    test_result,test_cases = span_metric(tmp_input_ids,temp_output,tmp_label,seq_steps,labels_lis,args.max_seq_length,model.tokenizer,ins_ids,return_case=True)
    # test_result = span_metric_for_embed(tmp_input_ids,temp_output,tmp_label,seq_steps,labels_lis,args.max_seq_length,model.word2idx)
    print("\n"+"="*5+" test result "+"="*5)
    for m,c in test_result.items():
        print("{} = {}%".format(m,c))
    print('-' * 40 + '\n')
    with open(os.path.join(args.output_dir,"test_result_{}.txt".format(data_name)),"a") as writer:
        writer.write(str(args) + '\n\n')
        for m,c in test_result.items():
            writer.write("{} = {}%\n".format(m,c))
        writer.write('-' * 40 + '\n')
    logger.info("save test result at {}".format(os.path.join(args.output_dir,"test_result.txt")))
    # save cases
    if args.save_case:
        with open(os.path.join(args.output_dir,"new_test_cases_{}_{}.json".format(data_name,args.model_name)),"w") as f:
            json.dump(test_cases,f)
        logger.info("save test cases at {}".format(os.path.join(args.output_dir,"new_test_cases_{}_{}.json".format(data_name,args.model_name))))

def main():
    parser = argparse.ArgumentParser()
    parser_model = parser.add_argument_group(title="ModelArguments")
    parser_train = parser.add_argument_group(title="DataTrainingArguments")
    
    parser_model.add_argument("--max_seq_length",type=int,default=512)
    parser_model.add_argument("--token_encoder_path",type=str,default="./pretrained/language_model/spanbert-base-cased",help="path to the pre-trained SpanBERT model")
    parser_model.add_argument("--bi_direction",action="store_true",help="Bi-LSTM or Uni-LSTM")
    parser_model.add_argument("--num_layer",type=int,default=1)
    parser_model.add_argument("--lstm_out_dim",type=int,default=150)
    parser_model.add_argument("--node_dim",type=int,default=600)
    parser_model.add_argument("--embed_path",type=str,default="./pretrained/word_embedding/glove.6B.300d.txt",help="path to the pre-trained word embedding")
    parser_model.add_argument("--model_name",type=str,default="McIN")

    parser_train.add_argument("--small_lr",type=float,default=2e-5)
    parser_train.add_argument("--media_lr",type=float,default=2e-4)
    parser_train.add_argument("--large_lr",type=float,default=1e-3)
    parser_train.add_argument("--eval_batch",type=int,default=4)
    parser_train.add_argument("--seed",type=int,default=44)
    parser_train.add_argument("--cuda",type=int,default=-1)
    parser_train.add_argument("--test_data_path",type=str,default=None)
    parser_train.add_argument("--test_context_path",type=str,default=None)
    parser_train.add_argument("--test_data_path_2",type=str,default=None)
    parser_train.add_argument("--test_context_path_2",type=str,default=None)
    parser_train.add_argument("--overwrite_cache",action="store_true")
    parser_train.add_argument("--output_dir",type=str,default="./output")
    parser_train.add_argument("--tensorboard",action="store_true")
    parser_train.add_argument("--shuffle",action="store_true")
    parser_train.add_argument("--reverse_shuffle",action="store_true")
    parser_train.add_argument("--save_best",action="store_true")
    parser_train.add_argument("--save_case",action="store_true")
    parser_train.add_argument("--quick_start",action="store_true",help="we provide our model parameters under the ./output, set `quick_start` to eval it directly")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    # set logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # set seed
    seed_torch(args.seed)

    # setup output_dir
    if not args.quick_start:
        template = "{}_{}_{}_{}_{}".format(args.small_lr,args.media_lr,args.large_lr,args.node_dim,args.lstm_out_dim)
        template += "_bi" if args.bi_direction else ""
        args.output_dir = os.path.join(args.output_dir,template)
    os.makedirs(args.output_dir,exist_ok=True)

    labels_lis = ["B-cause","I-cause","O"]
    emotion_lis = ["happy","surprise","disgust","anger","sad","fear"]  # pre-defined emotion categories (for RECCON-DD)
    emotion_dic = dict([(idx,emo) for idx,emo in enumerate(emotion_lis)])

    if "bert" in args.token_encoder_path:
        model_type = "bert"
    else:
        raise NotImplemented("use bert-base-cased or span-bert-base as default")

    # initialize model
    model = CausalGCN(token_enocder_path=args.token_encoder_path,embed_path=args.embed_path,emotion_dic=emotion_dic,cuda=args.cuda, seed=args.seed,
                lstm_out_dim=args.lstm_out_dim,node_dim=args.node_dim,bidirectional=args.bi_direction)

    # model.apply(show_parameters)
    logger.info("total param num : {}".format(count_parameters(model,verbose=True)))
    # model.weight_init()
    model.to(model.device)

    # test set
    tokenizer_path = args.token_encoder_path  ## use bert tokenizer
    test_data = (CauseDetectionDataset_GCN(args.test_data_path,args.test_context_path,tokenizer_path,labels_lis,model_type,args.max_seq_length,emotion_lis,args.overwrite_cache)
    if args.test_data_path is not None and args.test_context_path is not None else None
    )
    test_data_2 = (CauseDetectionDataset_GCN(args.test_data_path_2,args.test_context_path_2,tokenizer_path,labels_lis,model_type,args.max_seq_length,emotion_lis,args.overwrite_cache)
    if args.test_data_path_2 is not None and args.test_context_path_2 is not None else None
    )

    # load and test 
    model.load_state_dict(torch.load(os.path.join(args.output_dir,"{}.pt".format(args.model_name))))
    if args.test_data_path is not None:
        test_model(model,args,test_data,labels_lis,"dailydialog")
    if args.test_data_path_2 is not None:
        test_model(model,args,test_data_2,labels_lis,"iemocap")

if __name__ == "__main__":
    main()