'''
this script comes from https://github.com/declare-lab/RECCON/blob/main/eval_qa.py
we modified it to fit our setting (we hold different input and neglect those neg samples)
'''
import numpy as np
import json, os, logging, pickle, argparse
from .evaluate_squad import compute_f1

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set


def evaluate_results(text):
    partial_match_scores = []
    lcs_all = []
    impos1, impos2, impos3, impos4 = 0, 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0
    fscores, squad_fscores = [], [] # f1 for postive (valid) instances
    fscores_all, squad_fscores_all = [], [] # f1 for all instances
    
    impos_flag = False
    for i, key in enumerate(['correct_text', 'similar_text', 'incorrect_text']):
        for item in text[key]:
            if i==0:
                fscores_all.append(1)
                squad_fscores_all.append(1)
                if 'impossible' in item and text[key][item]['predicted'] == '':
                    impos_flag = True
                    impos2 += 1
                elif 'span' in item:
                    pos1 += 1
                    fscores.append(1)
                    squad_fscores.append(1)
                    
            elif i==1:
                if 'impossible' in item:
                    impos_flag = True
                    impos3 += 1
                    fscores_all.append(0)
                    squad_fscores_all.append(0)
                elif 'span' in item:
                    z = text[key][item]
                    if z['predicted'] != '':
                        longest_match = list(lcs(z['truth'], z['predicted']))[0]  ## calculate overlap
                        lcs_all.append(longest_match)
                        partial_match_scores.append(round(len(longest_match.split())/len(z['truth'].split()), 4))
                        pos2 += 1
                        r = len(longest_match.split())/len(z['truth'].split())
                        p = len(longest_match.split())/len(z['predicted'].split())
                        assert r <= 1.0 and p <= 1.0 
                        f = 2*p*r/(p+r)
                        fscores.append(f)
                        squad_fscores.append(compute_f1(z['truth'], z['predicted']))
                        fscores_all.append(f)
                        squad_fscores_all.append(compute_f1(z['truth'], z['predicted']))
                    else:
                        pos3 += 1
                        impos4 += 1
                        fscores.append(0)
                        squad_fscores.append(0)
                        fscores_all.append(0)
                        squad_fscores_all.append(0)
                                
            elif i==2:
                fscores_all.append(0)
                squad_fscores_all.append(0)
                z = text[key][item]
                if 'impossible' in item:
                    impos_flag = True
                    raise RuntimeError("if ground truth is '', it must belong to 'similar'")
                elif 'span' in item:
                    if z['predicted'] == '':
                        raise RuntimeError("if prediction is '', it must belong to 'similar'")
                    pos3 += 1
                    fscores.append(0)
                    squad_fscores.append(0)
                    
    total_pos = pos1 + pos2 + pos3
    ## if the datasets contain negative samples
    if impos_flag:
        assert impos2 != 0, "the model failed to predict all negative samples!"
        imr = impos2/(impos2+impos3)
        imp = impos2/(impos2+impos4)
        imf = 2*imp*imr/(imp+imr)
    else:
        imf = -0.01
    
    p1 = 'Postive Samples:'
    p2 = 'Exact Match: {}/{} = {}%'.format(pos1, total_pos, round(100*pos1/total_pos, 2))
    p3 = 'Partial Match: {}/{} = {}%'.format(pos2, total_pos, round(100*pos2/total_pos, 2))
    p4a = 'LCS F1 Score = {}%'.format(round(100*np.mean(fscores), 2))
    p4b = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores), 2))
    p5 = 'No Match: {}/{} = {}%'.format(pos3, total_pos, round(100*pos3/total_pos, 2))
    p6 = '\nNegative Samples:'
    p7 = 'Inv F1 Score = {}%'.format(round(100*imf, 2))  # negative F1
    # p7a = 'Inv Recall: {}/{} = {}%'.format(impos2, impos2+impos3, round(100*imr, 2))
    # p7b = 'Inv Precision: {}/{} = {}%'.format(impos2, impos2+impos4, round(100*imp, 2))
    
    p8 = '\nAll Samples:'
    p9a = 'LCS F1 Score = {}%'.format(round(100*np.mean(fscores_all), 2))
    p9b = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores_all), 2))

    p = '\n'.join([p1, p2, p3, p4a, p4b, p5, p6, p7, p8, p9a, p9b])

    result = {"Exact Match":round(100*pos1/total_pos, 2),
            "SQuAD F1":round(100*np.mean(squad_fscores), 2),
            "Inv F1":round(100*imf, 2),
            "SQuAD F1 (All)":round(100*np.mean(squad_fscores_all), 2),
            "Partial Match":round(100*pos2/total_pos, 2),
            "No Match":round(100*pos3/total_pos, 2)}

    return p,result