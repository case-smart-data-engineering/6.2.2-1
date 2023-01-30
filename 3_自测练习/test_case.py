#!/usr/bin/env python3

import os
from my_solution import get_probability


# 测试用例

os.system('my_solution.py')
with open("3_自测练习/cookbook_test.txt", "r", encoding="utf-8") as f:
    articles = f.readlines()

# 加入了参数为0.2的拉普拉斯平滑处理后，’青椒炒肉片‘出现的概率应为0.101
assert round(get_probability('青椒炒肉片'), 3) == 0.101
