#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .utils import read_data, estimate
from .model import CRF_Model

SENT_FILENAME = 'data/source_BIO_2014_cropus.txt'
TAG_FILENAME = 'data/target_BIO_2014_cropus.txt'
lines = 10000
sep_line = 8000

def train_and_eval(train_sent_data, train_tag_data, test_sent_data, test_tag_data):
    crf = CRF_Model()
    
    print("CRF Model training...")
    crf.train(train_sent_data, train_tag_data)
    print("CRF Model testing...")
    pred_tag_lists = crf.test(test_sent_data)
    pre, rec, f1 = estimate(pred_tag_lists, test_tag_data, print_result=True)



if __name__ == '__main__':
    sent_data, tag_data = read_data(SENT_FILENAME, TAG_FILENAME, lines)    

    train_sent_data = sent_data[:sep_line]
    train_tag_data = tag_data[:sep_line]
    test_sent_data = sent_data[sep_line:]
    test_tag_data = tag_data[sep_line:]
    
    train_and_eval(train_sent_data, train_tag_data, test_sent_data, test_tag_data) 
