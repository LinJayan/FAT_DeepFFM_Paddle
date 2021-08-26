### 基于Paddle复现FAT_DeepFFM模型
##### **论文名称**<a href="https://arxiv.org/pdf/1905.06336.pdff">FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine</a>

### **一、简介**

本文将深度场感知分解机(FAT-DeepFFM)与组合激发网络(DeepFFM)相结合，提出了一种增强场注意感知FTR模型，作为挤压激发网络(SENet)的增强版本，以突出其特征的重要性。

模型结构如下图：在DeepFFM的基础上，增加CENet作为Field的特征重要性选择。论文中使用到两种特征域交互的计算方式:Inner-product和Hadamard-product,后者的实验结果较好。

![](https://ai-studio-static-online.cdn.bcebos.com/21504be5f36145cba22c4db58014c555f8220129ce6d4729af61975a03e42b99)

### **二、复现精度**

**复现要求**：**AUC>0.8099**。论文中，在Criteo数据上的实验结果如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/02905ca3c6ec4e609f17ef081584268e74fd7c4e5afd4bbc8be1f483ac1a347e)

**本次复现结果**：**AUC=0.8037**。
（这里有点疑问的：在论文中的数据划分方式上，可能存在数据泄露问题！建议对复现精度和数据进行考虑。）

复现记录：

1.首先在PaddleRec官方提供的数据集上，训练集和测试集严格按照时间顺序划分的，不存在数据泄露问题，按照原论文的参数，实际复现AUC只能达到**0.8037**。
![](https://ai-studio-static-online.cdn.bcebos.com/eb9736057a3f44eba84177c922cb9ad192f216499a7a48488a05fb078f0aef7c)


2.按照随机划分的方式，将测试集和训练集中任意两个文件进行交换（存在数据泄露），这次在测试集上，测试的AUC竟然能达到**0.8058！！**如果将数据全部打乱随机切分训练集和测试集，精度应该是能达到0.8099，或许可以更大。
![](https://ai-studio-static-online.cdn.bcebos.com/0032b78f88f943dfb5f8eaf5c72d9add5b91e11b4f6d4867ae89c7d42eea917d)


### **三、数据集**
训练及测试数据集选用[PaddleRec](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)提供的Criteo数据集。

train set: 4400, 0000 条

test set: 184, 0617 条

该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  

### **四、环境依赖**
CPU、GPU均可，相应设置。

PaddlePaddle >= 2.1.2

Python >= 3.7

### **五、快速开始**

 ============================== Step 1,git clone 代码库 ==============================
 
git clone https://github.com/LinJayan/DIFM_Paddle.git

============================== Step 2 download data ==============================

Download  data

cd workpath//FAT_DeepFFM_Paddle/data && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz

tar xzvf slot_test_data_full.tar.gz
    
cd workpath//FAT_DeepFFM_Paddle/data && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz

tar xzvf slot_train_data_full.tar.gz

============================== Step 3, train model ==============================

启动训练、测试脚本 (需注意当前是否是 GPU 环境）

!cd workpath//FAT_DeepFFM_Paddle && sh run.sh config_bigdata.yaml

### **六、代码结构与详细说明**
```
├─models
   ├─ rank
        ├─fat_deepffm # DIFM模型代码
        ├──  data #样例数据
        ├── __init__.py
        ├── config.yaml # sample数据配置
        ├── config_bigdata.yaml # 全量数据配置
        ├── net.py # 模型核心组网（动静统一）
        ├── criteo_reader.py #数据读取程序
        ├── dygraph_model.py # 构建动态图
├─tools
├─README.md #文档
├─LICENSE #项目LICENSE
├─run.sh
```

### **七、模型信息**
**原论文重要参数和本项目复现参数对比**：参数保持一致！

|模型 | batch_size |lr |Sparse_dim |depth |FFM_dnn_size |other_dnn_size |activate |drop_out |reduction |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Fat_DeepFFM-original | 1000 | 0.0001 |10 |3 |[1600,1600] |[400,400] |ReLU |0.5 |1 |
| Fat_DeepFFM-paddle | 1000 | 0.0001 |10 |3 |[1600,1600] |[400,400] |ReLU  |0.5 |1 |