# 中文命名实体识别系统(Chinese Named Entity Recognition sys)

我们包括三个模型，包括CRF,BiLSTM和BERT

We include three model, including CRF,BiLSTM,BERT



# BILSTM

### 预处理数据(preprocess dataset)

```python
from BILSTM.util import combie_mini_data,combie_data_BIO
# 组合生成全集数据
combie_data_BIO(raw_data_file='./data/source.txt',raw_BIO_file='./data/BIO.txt')
# 生成mini数据
combie_mini_data(raw_data_file='./data/source.txt',raw_BIO_file='./data/BIO.txt',num=20000)
```

### 训练(train)

```python
from BILSTM.BILSTM import bilstm
# 设置超参数
model=bilstm(epoch=30,learning_rate=0.01,batch_size=64,print_step=5)
# 读取数据
model.read_data(test_data='./data/test.txt',dev_data='./data/dev.txt',train_data='./data/train.txt')
# 开始训练
model.train()
```



# BERT

### 使用crf模型(using crf model)

```python
from BERT.ner_crf import run_ner_crf
run_ner_crf()
```



### 使用softmax模型(using softmax)

```python
from BERT.ner_softmax import run_ner_softmax
run_ner_softmax()
```



### 使用span模型(using span model)

```python
from BERT.ner_span import run_ner_span
run_ner_span()
```

# CRF模型(CRF model)

### 训练(training )

```python
from CRF.model import CRF_Model
from CRF.utils import read_data

# 训练数据(train dataset)
sent_data, tag_data = read_data(sent_data='./data/source.txt',tag_file='./data/BIO.txt', lines=10000)    
sep_line = 8000
train_sent_data = sent_data[:sep_line]
train_tag_data = tag_data[:sep_line]
test_sent_data = sent_data[sep_line:]
test_tag_data = tag_data[sep_line:]

model = CRF_Model()
model.train(train_sent_data, train_tag_data)
```

测试(test)

```python
from CRF import ner as crf_ner
crf_ner(sentence)
```

