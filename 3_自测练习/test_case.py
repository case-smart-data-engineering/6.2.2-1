#!/usr/bin/env python3


from my_solution import get_probability


# 测试用例
def test_solution():
    sentence = '青椒炒肉片、青椒炒牛肉、青椒炒肉片、茄子炒肉末、茄子炒豆角'
    # 正确答案
    # 加入了参数为0.2的拉普拉斯平滑处理后，’青椒炒肉片‘出现的概率保留三位小数应为0.101
    correct_solution = 0.101
    # 程序求解结果
    result = round(get_probability(sentence, 3))
    assert correct_solution == result
