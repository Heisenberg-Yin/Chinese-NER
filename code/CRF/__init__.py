from .utils import read_data
from .model import CRF_Model

SENT_FILENAME = '../data/source_BIO_2014_cropus.txt'
TAG_FILENAME = '../data/target_BIO_2014_cropus.txt'
lines = 10000
sep_line = 8000

def ner(sentence):

    print("CRF Model Initializing...")
    sent_data, tag_data = read_data(SENT_FILENAME, TAG_FILENAME, lines)    

    train_sent_data = sent_data[:sep_line]
    train_tag_data = tag_data[:sep_line]
    
    model = CRF_Model()
    model.train(train_sent_data, train_tag_data)
    
    print("Predicting named entity...")
    pred_tag_lists = model.test([list(sentence)])
    print(f"Predicting result:\n{pred_tag_lists[0]}")

    return pred_tag_lists[0]
