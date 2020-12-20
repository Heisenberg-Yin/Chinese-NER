# --------------------------------------------------
# --------------使用BILSTM进行训练与测试-------------
# --------------------------------------------------

from BILSTM.BILSTM import bilstm
from BILSTM.util import combie_mini_data,combie_data_BIO

def BILSTM_train():
    # 设定模型超参数
    model=bilstm(epoch=30,learning_rate=0.01,batch_size=64,print_step=5)
    # 读取数据
    model.read_data(test_data='./data/test.txt',dev_data='./data/dev.txt',train_data='./data/train.txt')
    # 开始训练
    model.train()

def BILSTM_data_handle():
    # 生成组合数据
    # combie_data_BIO(raw_data_file='./data/source.txt',raw_BIO_file='./data/BIO.txt')
    # 生成mini数据
    combie_mini_data(raw_data_file='./data/source.txt',raw_BIO_file='./data/BIO.txt',num=20000)


# --------------------------------------------------
# ----------------使用BERT进行训练与测试-------------
# --------------------------------------------------

# 使用crf模型
from BERT.ner_crf import run_ner_crf
def bert_crf():
    run_ner_crf()


# 使用softmax模型
from BERT.ner_softmax import run_ner_softmax
def bert_softmax():
    run_ner_softmax()


# 使用span模型
from BERT.ner_span import run_ner_span
def bert_span():
    run_ner_span()

# --------------------------------------------------
# ----------------使用CRF进行训练与测试-------------
# --------------------------------------------------
from CRF import ner as crf_ner
from CRF.model import CRF_Model
from CRF.utils import read_data

def crf_train():
    sent_data, tag_data = read_data(sent_data='./data/source.txt',tag_file='./data/BIO.txt', lines=10000)    
    sep_line = 8000
    train_sent_data = sent_data[:sep_line]
    train_tag_data = tag_data[:sep_line]
    test_sent_data = sent_data[sep_line:]
    test_tag_data = tag_data[sep_line:]
    
    model = CRF_Model()
    model.train(train_sent_data, train_tag_data)

    return model

def crf_test(sentence):
    return crf_ner(sentence)


if __name__ == "__main__":
    # 使用自己想要的方法
    pass