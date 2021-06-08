import random


# 随机采样
def random_sampling(dataset, m):
    data = []
    for i in range(m):
        a = random.randint(0, len(dataset) - 1)
        data.append(dataset[a])
    return data
