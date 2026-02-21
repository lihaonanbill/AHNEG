from itertools import product


def enumerate_profiles(num_strategies):
    """
    枚举所有全局局势
    num_strategies: 一个列表，每个元素表示对应玩家的策略数量
    """
    profiles = list(product(*(range(1, n+1) for n in num_strategies)))
    return profiles



print(enumerate_profiles([1,2,3]))