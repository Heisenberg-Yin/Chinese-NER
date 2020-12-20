from .util import extend_maps

from .evaluate import bilstm_train_and_eval
from os.path import join
from codecs import open
from tqdm import tqdm

def build_corpus(data_file, make_vocab=True):
    """读取数据"""

    word_lists = []
    tag_lists = []
    with open(data_file, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        print('读取%s数据'%data_file)
        for line in tqdm(f.readlines()):
            if line != '\r\n':
                words=line.split()
                word, tag = words[0],words[1]
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回self.word2id和self.tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    print('make_vocab')
    for list_ in tqdm(lists):
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

class bilstm:
    def __init__(self,epoch=30,learning_rate=0.01,batch_size=64,print_step=5):
        self.epoch=epoch
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.print_step=print_step

    def read_data(self,test_data:str,dev_data:str,train_data:str):
        self.train_word_lists, self.train_tag_lists, self.word2id, self.tag2id = \
        build_corpus(train_data)
        self.dev_word_lists, self.dev_tag_lists = build_corpus(test_data, make_vocab=False)
        self.test_word_lists, self.test_tag_lists = build_corpus(test_data, make_vocab=False)

    def train(self):
        # 训练评估BI-LSTM模型
        print("BILSTM开始训练！")
        # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
        bilstm_word2id, bilstm_tag2id = extend_maps(self.word2id, self.tag2id, for_crf=False)
        lstm_pred = bilstm_train_and_eval(
            (self.train_word_lists, self.train_tag_lists),
            (self.dev_word_lists, self.dev_tag_lists),
            (self.test_word_lists, self.test_tag_lists),
            bilstm_word2id, bilstm_tag2id,
            crf=False
        )
    
    