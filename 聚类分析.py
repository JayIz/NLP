import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
from matplotlib.font_manager import FontProperties
from pandas import read_csv
from scipy.cluster.hierarchy import dendrogram,  ward
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
from sklearn.manifold import MDS
import nltk
from nltk.cluster.kmeans import KMeansClusterer

#加载数据包及数据整理

class MaxProbCut:
    def __init__(self):
        self.word_dict = {}  # 记录概率,1-gram
        self.word_dict_count = {}  # 记录词频,1-gram
        self.trans_dict = {}  # 记录概率,2-gram
        self.trans_dict_count = {}  # 记录词频,2-gram
        self.max_wordlen = 0  #词的最长长度
        self.all_freq = 0  # 所有词的词频总和,1-gram
        word_count_path = "./model/word_dict.model"
        word_trans_path = './model/trans_dict.model'
        self.init(word_count_path, word_trans_path)

    # 加载词典
    def init(self, word_count_path, word_trans_path):
        self.word_dict_count = self.load_model(word_count_path)
        self.all_freq = sum(self.word_dict_count.values())  # 所有词的词频
        self.max_wordlen = max(len(key) for key in self.word_dict_count.keys())
        for key in self.word_dict_count:
            self.word_dict[key] = math.log(self.word_dict_count[key] / self.all_freq)
        #计算转移概率
        Trans_dict = self.load_model(word_trans_path)
        for pre_word, post_info in Trans_dict.items():
            for post_word, count in post_info.items():
                word_pair = pre_word + ' ' + post_word
                self.trans_dict_count[word_pair] = float(count)
                if pre_word in self.word_dict_count.keys():
                    self.trans_dict[key] = math.log(count / self.word_dict_count[pre_word])  # 取自然对数，归一化
                else:
                    self.trans_dict[key] = self.word_dict[post_word]

    #加载预训练模型
    def load_model(self, model_path):
        f = open(model_path, 'rb')
        a = f.read()
        word_dict = eval(a)
        f.close()
        return word_dict

    # 估算未出现的词的概率,根据beautiful data里面的方法估算，平滑算法
    def get_unknow_word_prob(self, word):
        return math.log(1.0 / (self.all_freq ** len(word)))

    # 获取候选词的概率
    def get_word_prob(self, word):
        if word in self.word_dict.keys():  # 如果字典包含这个词
            prob = self.word_dict[word]
        else:
            prob = self.get_unknow_word_prob(word)
        return prob

    #获取转移概率
    def get_word_trans_prob(self, pre_word, post_word):
        trans_word = pre_word + " " + post_word
        
        if trans_word in self.trans_dict_count.keys():
            trans_prob = math.log(self.trans_dict_count[trans_word] / self.word_dict_count[pre_word])
        else:
            trans_prob = self.get_word_prob(post_word)
        return trans_prob

    # 寻找node的最佳前驱节点，方法为寻找所有可能的前驱片段
    def get_best_pre_node(self, sentence, node, node_state_list):
        # 如果node比最大词长小，取的片段长度以node的长度为限
        max_seg_length = min([node, self.max_wordlen])
        pre_node_list = []  # 前驱节点列表

        # 获得所有的前驱片段，并记录累加概率
        for segment_length in range(1, max_seg_length + 1):
            segment_start_node = node - segment_length
            segment = sentence[segment_start_node:node]  # 获取片段
            pre_node = segment_start_node  # 取该片段，则记录对应的前驱节点
            if pre_node == 0:
                # 如果前驱片段开始节点是序列的开始节点，
                # 则概率为<S>转移到当前词的概率
                segment_prob = self.get_word_trans_prob("<BEG>", segment)
            else:  # 如果不是序列开始节点，按照二元概率计算
                # 获得前驱片段的前一个词
                pre_pre_node = node_state_list[pre_node]["pre_node"]
                pre_pre_word = sentence[pre_pre_node:pre_node]
                segment_prob = self.get_word_trans_prob(pre_pre_word, segment)

            pre_node_prob_sum = node_state_list[pre_node]["prob_sum"]  # 前驱节点的概率的累加值
            # 当前node一个候选的累加概率值
            candidate_prob_sum = pre_node_prob_sum + segment_prob
            pre_node_list.append((pre_node, candidate_prob_sum))

        # 找到最大的候选概率值
        (best_pre_node, best_prob_sum) = max(pre_node_list, key=lambda d: d[1])

        return best_pre_node, best_prob_sum


    #切词主函数
    def cut_main(self, sentence):
        sentence = sentence.strip()
        # 初始化
        node_state_list = []  # 记录节点的最佳前驱，index就是位置信息
        # 初始节点，也就是0节点信息
        ini_state = {}
        ini_state["pre_node"] = -1  # 前一个节点
        ini_state["prob_sum"] = 0  # 当前的概率总和
        node_state_list.append(ini_state)
        # 字符串概率为2元概率， P(a b c) = P(a|<S>)P(b|a)P(c|b)
        # 逐个节点寻找最佳前驱节点
        for node in range(1, len(sentence) + 1):
            # 寻找最佳前驱，并记录当前最大的概率累加值
            (best_pre_node, best_prob_sum) = self.get_best_pre_node(sentence, node, node_state_list)

            # 添加到队列
            cur_node = {}
            cur_node["pre_node"] = best_pre_node
            cur_node["prob_sum"] = best_prob_sum
            node_state_list.append(cur_node)

        # step 2, 获得最优路径,从后到前
        best_path = []
        node = len(sentence)  # 最后一个点
        best_path.append(node)
        while True:
            pre_node = node_state_list[node]["pre_node"]
            if pre_node == -1:
                break
            node = pre_node
            best_path.append(node)
        best_path.reverse()

        # step 3, 构建切分
        word_list = []
        for i in range(len(best_path) - 1):
            left = best_path[i]
            right = best_path[i + 1]
            word = sentence[left:right]
            word_list.append(word)

        return word_list

    def cut(self, sentence):
        return self.cut_main(sentence)



## 设置字体和 设置pandas显示方式
font=FontProperties(fname = "C:/Windows/Fonts/AdobeSongStd-Light.otf",size=14)

pd.set_option("display.max_rows",8)
pd.options.mode.chained_assignment = None  # default='warn'

## 读取停用词和需要的词典
stopword = read_csv("my_stop_words.txt",header=None,names = ["Stopwords"])

RedDream = read_csv("red.txt",header=None,names = ["Reddream"])


#删除空白行和不需要的段，并重新设置索引
np.sum(pd.isnull(RedDream))  #查看数据是否有空白的行，如有则删除
indexjuan = RedDream.Reddream.str.contains("^第+.+卷") # 删除卷数据，使用正则表达式，包含相应关键字的索引
RedDream = RedDream[~indexjuan].reset_index(drop=True) ## 删除不需要的段，并重新设置索引


## 找出每一章节的头部索引和尾部索引
## 每一章节的标题
indexhui = RedDream.Reddream.str.match("^第+.+回")
chapnames = RedDream.Reddream[indexhui].reset_index(drop=True)

## 处理章节名，按照空格分割字符串
chapnamesplit = chapnames.str.split(" ").reset_index(drop=True)

## 建立保存数据的数据表
Red_df=pd.DataFrame(list(chapnamesplit),columns=["Chapter","Leftname","Rightname"])
## 添加新的变量
Red_df["Chapter2"] = np.arange(1,121)
Red_df["ChapName"] = Red_df.Leftname+","+Red_df.Rightname
## 每章的开始行（段）索引
Red_df["StartCid"] = indexhui[indexhui == True].index
## 每章的结束行数
Red_df["endCid"] = Red_df["StartCid"][1:len(Red_df["StartCid"])].reset_index(drop = True) - 1
Red_df["endCid"][[len(Red_df["endCid"])-1]] = RedDream.index[-1]
## 每章的段落长度
Red_df["Lengthchaps"] = Red_df.endCid - Red_df.StartCid
Red_df["Artical"] = "Artical"

## 每章节的内容
for ii in Red_df.index:
    ## 将内容使用""连接
    chapid = np.arange(Red_df.StartCid[ii]+1,int(Red_df.endCid[ii]))
    ## 每章节的内容替换掉空格
    Red_df["Artical"][ii] = "".join(list(RedDream.Reddream[chapid])).replace("\u3000","")
## 计算某章有多少字
Red_df["lenzi"] = Red_df.Artical.apply(len)


## 对红楼梦全文进行分词
## 数据表的行数
row,col = Red_df.shape
#加载分词方法
cuter = MaxProbCut()
## 预定义列表
Red_df["cutword"] = "cutword"
for ii in np.arange(row):
    ## 分词
    cutwords = list(cuter.cut(Red_df.Artical[ii]))
    ## 去除长度为1的词
    cutwords = pd.Series(cutwords)[pd.Series(cutwords).apply(len)>1]
    ## 去停用此
    cutwords = cutwords[~cutwords.isin(stopword)]
    Red_df.cutword[ii] = cutwords.values


## 保存数据
Red_df.to_json("Red_dream_data.json")

#打印数据
with open('Red_dream_data.json') as f_obj:
    print(json.load(f_obj))

#使用夹角余弦距离进行k均值聚类

articals = []
for cutword in Red_df.cutword:
    articals.append(" ".join(cutword))
## 构建语料库，并计算文档－－词的TF－IDF矩阵
vectorizer = CountVectorizer()
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(articals)

## tfidf 以稀疏矩阵的形式存储，将tfidf转化为数组的形式,文档－词矩阵
dtm = tfidf.toarray()

## 使用夹角余弦距离进行k均值聚类
kmeans = KMeansClusterer(num_means=2,       #聚类数目
                         distance=nltk.cluster.util.cosine_distance,  #夹角余弦距离
                         )
kmeans.cluster(dtm)

## 聚类得到的类别
labpre = [kmeans.classify(i) for i in dtm]
kmeanlab = Red_df[["ChapName","Chapter"]]
kmeanlab["cosd_pre"] = labpre
kmeanlab


## 查看每类有多少个分组
count = kmeanlab.groupby("cosd_pre").count()

## 将分类可视化
count.plot(kind="barh",figsize=(6,5))
for xx,yy,s in zip(count.index,count.ChapName,count.ChapName):
    plt.text(y =xx-0.1, x = yy+0.5,s=s)
plt.ylabel("cluster label")
plt.xlabel("number")
plt.show()


#MDS降维

mds = MDS(n_components=2,random_state=123)
coord = mds.fit_transform(dtm)
plt.figure(figsize=(8,8))
plt.scatter(coord[:,0],coord[:,1],c=kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii,0]+0.02,coord[ii,1],s = Red_df.Chapter2[ii])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means MDS")
plt.show()

# 层次聚类

labels = Red_df.Chapter.values
cosin_matrix = squareform(pdist(dtm,'cosine'))#计算每章的距离矩阵
ling = ward(cosin_matrix)  ## 根据距离聚类

## 聚类结果可视化
fig, ax = plt.subplots(figsize=(10, 15)) # 设置大小
ax = dendrogram(ling,orientation='right', labels=labels);
plt.yticks(FontProperties = font,size = 8)
plt.title("《红楼梦》各章节层次聚类",FontProperties = font)
plt.tight_layout()
plt.show()









