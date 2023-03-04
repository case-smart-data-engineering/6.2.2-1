#!/usr/bin/env python3


# 待测试程序
import jieba
import re


def cut(string): return list(jieba.cut(string))
TOKEN = cut('青椒炒肉片、青椒炒牛肉、青椒炒肉片、茄子炒肉末、茄子炒豆角')
# print(TOKEN)
# ['青椒', '炒', '肉片', '青椒', '炒', '牛肉', '青椒', '炒', '肉片', '茄子', '炒', '肉末', '茄子', '炒', '豆角']

# 对TOKEN中的词进行频数统计，结果存入words_count
from collections import Counter
words_count = Counter(TOKEN)
# print(words_count)
# Counter({'炒': 5, '青椒': 3, '肉片': 2, '茄子': 2, '牛肉': 1, '肉末': 1, '豆角': 1})

# 对token里面相邻的两个词进行组合，经行频数统计，结果存入words_count_2
TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]
words_count_2 = Counter(TOKEN_2_GRAM)
# print(TOKEN_2_GRAM)
# ['青椒炒', '炒肉片', '肉片青椒', '青椒炒', '炒牛肉', '牛肉青椒', '青椒炒', '炒肉片', '肉片茄子', '茄子炒', '炒肉末', '肉末茄子', '茄子炒']
# print(words_count_2)
# Counter({'青椒炒': 3, '炒肉片': 2, '茄子炒': 2, '肉片青椒': 1, '炒牛肉': 1, '牛肉青椒': 1, '肉片茄子': 1, '炒肉末': 1, '肉末茄子': 1})

# 构建2—gram模型
v = len(words_count.keys())  # 单个分词总的类别数
m = len(words_count_2.keys())  # 两个连着的分词总的类别数
# 为防止经常出现零概率问题，这里计算概率时采用了拉普拉斯平滑处理，平滑参数为0.2
# 计算单个词出现的概率
def prob_1(word, sig=0.2):
    return (words_count[word] + sig) / (len(TOKEN)+sig*v)


# 计算两个组合的词出现的概率
def prob_2(word1, word2, sig=0.2):
    return (words_count_2[word1+word2] + sig) / (len(TOKEN_2_GRAM) + sig*m)


# 计算某个句子的概率（公式在算法演示的demo.gif中）
def get_probability(sentence):
    words = cut(sentence)
    sentence_prob = 1
    for i, word in enumerate(words[:-1]):
        next_word = words[i+1]
        probability_1 = prob_1(next_word)
        probability_2 = prob_2(word, next_word)
        sentence_prob *= (probability_2 / probability_1)
    sentence_prob *= probability_1
    return sentence_prob


if __name__ == '__main__':
    pass
