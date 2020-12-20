#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

DIR = os.path.dirname(__file__)


# Read word and tag files
def read_data(word_file, tag_file=None, lines=10000):
    word_lists, tag_lists = [], []
    
    with open(os.path.join(DIR, word_file), 'r') as f:
        for i in range(lines):
            line = f.readline()
            if not line.strip():
                continue
            word_lists.append(line.strip().split(' '))

    if tag_file is not None:
        with open(os.path.join(DIR, tag_file), 'r') as f:
            for i in range(lines):
                line = f.readline()
                if not line.strip():
                    continue
                tag_lists.append(line.strip('\n').split(' '))

    return word_lists, tag_lists


# Tools for CRF Model
# Get features for single word
# Features defined in the function:
#   previous word, current word, next word
#   previous word + current word, current word + next word
def word2features(sent, i):
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent)-1) else sent[i+1]

    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features

# Tools for CRF Model
# Get features for a list of sentence 
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# Calculate and print Precision, Recall and F1
def estimate(pred_tags, std_tags, print_result=False):
    std_tags = [i for ls in std_tags for i in ls]
    pred_tags = [i for ls in pred_tags for i in ls]
    tags = set(std_tags)
    tp, fp, fn = {}, {}, {}
    for t in tags:
        tp[t] = fp[t] = fn[t] = 0

    for i in range(len(std_tags)):
        if pred_tags[i] == std_tags[i]:
            tp[std_tags[i]] += 1
        else:
            fp[pred_tags[i]] += 1
            fn[std_tags[i]] += 1

    pre, rec, f1 = {'avg': 0}, {'avg': 0}, {'avg': 0}
    for t in tags:
        std_size = tp[t] + fn[t]
        pre[t] = tp[t] / (tp[t] + fp[t])
        pre['avg'] += pre[t] * std_size
        rec[t] = tp[t] / std_size
        rec['avg'] += rec[t] * std_size
        f1[t] = 2. * pre[t] * rec[t] / (pre[t] + rec[t] + 1e-20)
        f1['avg'] += f1[t] * std_size
    
    total = len(std_tags)
    pre['avg'] /= total
    rec['avg'] /= total
    f1['avg'] /= total

    if print_result:
        print("Tags\t\tPrecision\tRecall\t\tF1\t\tSupport")
        for t in tags:
            print(f"{t}\t\t{pre[t]:.4f}\t\t{rec[t]:.4f}\t\t{f1[t]:.4f}\t\t{tp[t] + fn[t]}")
        print(f"\nAvg\t\t{pre['avg']:.4f}\t\t{rec['avg']:.4f}\t\t{f1['avg']:.4f}\t\t{total}")
        std_O_size = tp['O'] + fn['O']
        pre_avg_without_O = ((pre['avg'] * total) - (pre['O'] * std_O_size)) / (total - std_O_size)
        rec_avg_without_O = ((rec['avg'] * total) - (rec['O'] * std_O_size)) / (total - std_O_size)
        f1_avg_without_O = ((f1['avg'] * total) - (f1['O'] * std_O_size)) / (total - std_O_size)
        print(f"Avg without O\t{pre_avg_without_O:.4f}\t\t{rec_avg_without_O:.4f}\t\t{f1_avg_without_O:.4f}\t\t{total - std_O_size}")

    return pre, rec, f1
