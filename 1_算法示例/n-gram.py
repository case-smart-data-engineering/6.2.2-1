import jieba
import re


def cut(string): return list(jieba.cut(string))


# jieba.cut返回迭代器，这里可用jieba.lcut(string)代替list(jieba.cut(string))
# cut('青椒炒肉片')  # ['青椒', '炒', '肉片']

# 自己构建语料库
with open("1_算法示例/cookbook.txt", "r", encoding="utf-8") as f:
    articles = f.readlines()
# 切词，清洗，对读取的语料去标点、空格
def token(string):
    return re.findall('\w+', string)


articles_clean = [''.join(token(str(a))) for a in articles]
# 对文本切词处理，存入TOKEN
TOKEN = []
for i, article in enumerate(articles_clean):
    if i % 10000 == 0: print(i)
    TOKEN += cut(article)
# print(TOKEN)
# ['咕噜', '肉', '梅干菜', '扣肉', '木须肉', '鱼香肉丝', '东坡肉', '荷叶',...]


# 对TOKEN中的词进行频数统计，结果存入words_count
from collections import Counter
words_count = Counter(TOKEN)
# print(words_count)
# Counter({'牛肉': 14, '凉拌': 7, '肉': 6, '土豆': 5, '拌': 5,...})


# 对token里面相邻的两个词进行组合，经行频数统计，结果存入words_count_2
TOKEN_2_GRAM = [''.join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]
words_count_2 = Counter(TOKEN_2_GRAM)
# print(TOKEN_2_GRAM)
# ['咕噜肉', '肉梅干菜', '梅干菜扣肉', '扣肉木须肉', '木须肉鱼香肉丝'...]
# print(words_count_2)
# Counter({'剁椒': 4, '咕噜肉': 2, '手撕': 2, '蒜蓉蒸': 2, '蒸茄子': 2, '茄子土豆': 2, '土豆烧茄子': 2,...})

# 构建2—gram模型
v = len(words_count.keys())  # 单个分词总的类别数
m = len(words_count_2.keys())  # 两个连着的分词总的类别数

# 为防止经常出现零概率问题，这里计算概率时采用了拉普拉斯平滑处理
# 计算单个词出现的概率
def prob_1(word, sig=0.2):
    return (words_count[word] + sig) / (len(TOKEN)+sig*v)


# 计算两个组合的词出现的概率
def prob_2(word1, word2, sig=0.2):
    return (words_count_2[word1+word2] + sig) / (len(TOKEN_2_GRAM) + sig*m)


# 计算某个句子的概率（公式在算法演示的gif图中的2—gram模型中）
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

# 计算’长沙臭豆腐‘出现的概率，保留三位小数
print(round(get_probability('长沙臭豆腐'), 3))
