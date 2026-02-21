"""
function:
    Based on Game.py, calculate the transition matrices under different asynchronous schemes
    

"""

from Game import *
from scipy.linalg import khatri_rao
from math import lcm
from itertools import product



def find_next_best_asynchronous_pofile(profile, games, payoffs, num_strategies, asynchronous_scheme):
    """
    函数功能描述：
        find_next_best_asynchronous_pofile() 是基于 Game.find_next_best_profile() 进行修改，
        添加功能，根据 asynchronous_scheme 确定哪些玩家会更新，哪些玩家保持策略不变。
        asynchronous_scheme 是一个长度等于玩家数的列表，元素为1或2:
            1 表示该玩家允许更新
            2 表示该玩家保持原有策略
    
    参数：
    :param profile: tuple，当前局势
    :param games: dict，超博弈结构
    :param payoffs: dict，子博弈收益
    :param num_strategies: list，每个玩家的策略数
    :param asynchronous_scheme: list，元素为1或2，表示每个玩家是否允许更新
    
    返回：
    :return: tuple，下一步最优局势
    """
    current_payoffs = calculate_global_payoffs(profile, games, payoffs, num_strategies)
    new_profile = list(profile)

    for player in range(len(num_strategies)):
        if asynchronous_scheme[player] == 2:
            continue  # 不更新，保持原策略

        best_strategy = profile[player]
        best_payoff = current_payoffs[player]
        
        # 尝试改变该玩家的策略
        for s in range(1, num_strategies[player] + 1):
            if s == profile[player]:
                continue
            test_profile = list(profile)
            test_profile[player] = s
            test_payoffs = calculate_global_payoffs(test_profile, games, payoffs, num_strategies)
            
            if test_payoffs[player] > best_payoff:
                best_payoff = test_payoffs[player]
                best_strategy = s
            elif test_payoffs[player] == best_payoff:
                best_strategy = max(best_strategy, s)
        
        new_profile[player] = best_strategy

    return tuple(new_profile)

# 求出某个异步模式下的转移矩阵
def get_asynchronous_transition_matrix(games, payoffs, num_strategies, asynchronous_scheme):
    """
    函数功能：
        基于Game.get_transition_matrix()
        构造某一个异步更新模式下的状态转移矩阵
    
    参数：
    :param games: dict
    :param payoffs: dict
    :param num_strategies: list，每个玩家所拥有的策略数量
    :param asynchronous_scheme: list，允许更新的玩家编号
    
    返回：
    :return: list，下一步转移到的局势编号索引
    """
    all_profiles = enumerate_profiles(num_strategies)
    profile_to_index = { profile: idx+1 for idx, profile in enumerate(all_profiles) }
    next_profile_list = []

    for profile in all_profiles:
        next_profile = find_next_best_asynchronous_pofile(profile, games, payoffs, num_strategies, asynchronous_scheme)
        index_of_next = profile_to_index[next_profile]
        next_profile_list.append(index_of_next)

    return next_profile_list



# 求出多个异步模式下的转移矩阵
def get_asynchronous_transition_matrices(games, payoffs, num_strategies, asynchronous_schemes):
    """
    函数功能：
        基于get_asynchronous_transition_matrix()
        构造不同异步更新下模式下的状态转移矩阵
    
    参数：
    :param games: dict
    :param payoffs: dict
    :param num_strategies: list，每个玩家所拥有的策略数量
    :param asynchronous_scheme: list，不同的异步更新模式，
        如对于一个三个玩家的博弈[[1,1,1],[1,2,1]],表示全部更新和(1,3)更新，2不更新的情况
    
    返回：
    :return: list，
        下一步转移到的局势编号索引为该list中的一个元素
    """

    all_profiles = enumerate_profiles(num_strategies)
    profile_to_index = { profile: idx+1 for idx, profile in enumerate(all_profiles) }  # 1-based

    all_transition_matrices = []

    for scheme in asynchronous_schemes:
        next_profile_list = []
        for profile in all_profiles:
            next_profile = find_next_best_asynchronous_pofile(profile, games, payoffs, num_strategies, scheme)
            index_of_next = profile_to_index[next_profile]
            next_profile_list.append(index_of_next)
        all_transition_matrices.append(next_profile_list)

    return all_transition_matrices


# 求出状态相关异步的转移矩阵，本质上(按照论文里的方法)是1.给定异步模式，2.某个局势，求出下一步局势
# 其实我这里是可以用STP的，但因为STP函数定义较晚，就没有用，而是采用了笨办法
def get_state_driven_transition_matrix(games, payoffs, num_strategies, asyn_choice):
    """
    给出状态相关异步转移矩阵
    模型(9)中的 overline{L}
    
    参数：
    :param games: dict
    :param payoffs: dict
    :param num_strategies: list
    :param asyn_choice: list
        每一个元素是一个整数，对应异步方案的字典序编号
        字典序由 (1,2) 的 n 元笛卡尔积排列而成
    返回：
    :return: list，下一步转移到的局势编号
    """
    n_players = len(num_strategies)
    
    # 先列出所有可能的异步方案
    all_schemes = list(product([1,2], repeat=n_players))  # 字典序
    all_profiles = enumerate_profiles(num_strategies)
    profile_to_index = { profile: idx+1 for idx, profile in enumerate(all_profiles) }
    next_profile_list = []
    

    for idx, profile in enumerate(all_profiles):
        scheme_idx = asyn_choice[idx]  # 比如 3
        asynchronous_scheme = all_schemes[scheme_idx - 1]  # 因为字典序编号是从 1 开始
        next_profile = find_next_best_asynchronous_pofile(
            profile, games, payoffs, num_strategies, asynchronous_scheme
        )
        index_of_next = profile_to_index[next_profile]
        next_profile_list.append(index_of_next)
        
    return next_profile_list


# 从每个玩家的异步概率计算各个异步模式极其概率权重
def get_weights_and_all_asynchronous_schemes(probability_update):
    """
    功能：
    :function:利用每个玩家更新的概率计算出各个异步模式的权重和所有的异步模式
        1.首先利用probability_update的长度获得玩家的数量，
        2.给出所有的异步模式
            用1表示玩家更新，用2表示玩家保持上一时刻的策略，用字典序排列所有的异步模式
            举个例子如果len(probability_update) = 2, 那么输出的所有异步更新模式应该是
            [
                [1,1],
                [1,2],
                [2,1],
                [2,2]
            ]
        3.计算每个异步模式对应的概率权重
            例如，如果输入的概率是[0.2,0.4]
            那么应该输出[0.2x0.4,0.2x(1-0.4),(1-0.2)x0.4,(1-0.2)x(1-0.4)]

    
    
    参数：
    :param probability_update: list
        表示每个玩家更新的概率
        例如[0.2,0.5,0.6]表示一个博弈中一共有3个玩家，
        第一个玩家更新和保持上一时刻的策略概率分别是0.2，0.8，第二个玩家分别是0.5，0.5，第三个玩家分别是0.6，0.4
    
    返回：
    :return: 
        1.所有的异步更新模式
        2.概率权重，用列表表示
             
    """
    n = len(probability_update)

    # 所有异步更新模式（1 表示更新，2 表示保持）
    async_schemes = list(product([1, 2], repeat=n))

    # 对应的概率权重
    weights = []
    for scheme in async_schemes:
        prob = 1.0
        for i, choice in enumerate(scheme):
            p = probability_update[i]
            prob *= p if choice == 1 else (1 - p)
        weights.append(round(prob, 2))

    return [ [list(s) for s in async_schemes], weights ]






if __name__ == "__main__":


    # 你给的例子
    games = {
    'game1': ['1', '2', '3'],
    'game2': ['3', '4'],
    'game3': ['4', '5'],
    'game4': ['3', '5']



    }
    
    # 这里 payoffs 你需要补充真实数值
    payoffs = {
        # pdg  初始资源 10; 每人每次只能贡献0或10; 乘数因子1.5 
        'game1': 
                [
            [1, 1, 1],       # CCC
            [0.33, 0.33, 1.33], # CCN
            [0.33, 1.33, 0.33], # CNC
            [-0.33, 0.67, 0.67],# CNN
            [1.33, 0.33, 0.33], # NCC
            [0.67, -0.33, 0.67],# NCN
            [0.67, 0.67, -0.33],# NNC
            [0, 0, 0]           # NNN
        ],
        'game2': [
            [4, 4],     # 11
            [4, 0],     # 12
            [0, 4],     # 21
            [10, 10]      # 22
        ],
        'game3': [
            [4, 4],     # 11
            [4, 0],     # 12
            [0, 4],     # 21
            [10, 10]      # 22
        ],
        'game4': [
            [4, 4],     # 11
            [4, 0],     # 12
            [0, 4],     # 21
            [10, 10]      # 22
        ]
    }
    num_strategies = [2,2,2,2,2]


    #  如果直接使用STP的话只需要把同步情况下TransitionMatrix和下面的异步模式各自对应的逻辑矩阵相乘即可，就像论文里所描述的那样
    # TransitionMatrix = get_state_driven_transition_matrix(
    #     games, payoffs, num_strategies, 
    #     [1,1,1,1,
    #      2,1,1,1,
    #      1,8,16,4,
    #      2,1,1,6
    #      ]
    #     )
    
    

    # TransitionMatrix = get_asynchronous_transition_matrix(
    #     games, payoffs, num_strategies, [1,1,1,1,1]
        
        
    #     )

    # TransitionMatrix = get_transition_matrix(games, payoffs, num_strategies)
    
    # print(TransitionMatrix)


    # temp= []
    # temp.append(TransitionMatrix)
    # TransitionMatrix = temp

    # find_all_cycles(TransitionMatrix)

    # # draw_directed_graph(TransitionMatrix)

    # predecessors = find_predecessors([25,32],TransitionMatrix)
    # print(predecessors)
    # print(find_max_mus([25,32], predecessors, num_strategies))

    print(get_weights_and_all_asynchronous_schemes([0.3,0.2,0.1]))


    