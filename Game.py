"""
initial_param:

1. hypergraph structure
2. payoff matrix corresponding to each hyper edge
3. num_strategies


e.g.
1.
规则如下:
    每个hyperedge中的玩家编号必须升序排列

games = {
    'game1': ['1', '2', '3'],
    'game2': ['3', '4'],
    'game3': ['4', '5', '6']
}

2.

规则如下:
    每个子博弈的payoff 数据结构设计如下,将每个hyperedge中的profile按字典序升序排列,然后再列表中按照字典序给出每个玩家的收益
    以 下面payoff中的'game2'为例,假设每个玩家的策略数为2, 那么[[a,b],[c,d],[e,f],[g,h]]就分别表示了玩家3,4分别取策略11,12,21,22
    时获得的收益


payoff matrix
payoffs = {
    'game1':
    'game2': [[a,b],[c,d],[e,f],[g,h]],
    'game3': 
}

"""



"""
prompt:
    编写一个python程序，在我设计的超网络演化博弈中，计算每一个局势对应的下一个最优局势    
    input:包括两个参数
        games,payoffs
    output：
        输出最优局势

    1.首先定义一个玩家局势的字典序排列顺序，下面举例说明
        比如有两个玩家，每个玩家有两个策略分别用1，2表示
        那么[[11],[12],[21],[22]],就是这个博弈中所有局势的字典序表示，其中[ab]表示玩家1取策略a，玩家2取策略b
    
    2.这里的最优局势是这样计算的，
            首先利用games和payoffs得出每一个局势对应的每个玩家的收益
                每个玩家可能会参与多个博弈，要将这些博弈的收益进行累加
                这个收益用一个列表记录
                [[a,b],[c,d],[e,f],[g,h]],还是利用上述两个玩家的例子说明，[a,b]表示玩家1，2分别取策略1，1时分别获得收益a,b
            然后，对于每一个局势中每一个玩家的最优收益，
                以玩家1，局势11为例，
                令其余玩家在该局势中对应策略保持不变，
                    在这里也就是使玩家2保持策略1不变
                然后比较玩家1取不同策略时对应局势的收益，
                    在这里就是比较a和e的收益
                然后取收益最大的策略作为当前局势下该玩家的最优策略，如果最优策略有多个，取策略值最大的那个
                    在这里就是比较a,e然后取策略1或2
                    如果a=e,则取2
            由此可以得到所有局势对应的下一步最优局势
                在本例中可以得到[[ij],[kl],[mn],[op]]

"""


from StateTransitionGraph import *

from itertools import product

def enumerate_profiles(num_strategies):
    """
    枚举所有全局局势
    num_strategies: 一个列表，每个元素表示对应玩家的策略数量
    """
    profiles = list(product(*(range(1, n+1) for n in num_strategies)))
    return profiles

def get_subgame_payoff(subgame_payoffs, subgame_profile, subgame_players, num_strategies):
    """
    给定一个局势（子博弈的），返回该局势下所有玩家的收益
    subgame_players: 子博弈涉及的玩家编号（字符串）
    num_strategies: 所有玩家的策略数列表

    原理：将一个多进制编码转化一维下标

    """
    idx = 0
    base = 1
    for s, p in zip(reversed(subgame_profile), reversed(subgame_players)):
        player_idx = int(p) - 1  # 全局玩家编号
        idx += (s - 1) * base
        base *= num_strategies[player_idx]
    return subgame_payoffs[idx]

def calculate_global_payoffs(profile, games, payoffs, num_strategies):
    """
    profile: (s1, s2, s3, ...) 表示每个玩家的策略，指某一个全局局势
    计算全局局势下每位玩家的收益
    """
    player_payoffs = [0 for _ in range(len(num_strategies))]


    """
    遍历超网络
    比如第一次 game_name = 'game1', players = ['1','2','3']
    """
    for game_name, players in games.items():
        # 取出该局势在这个子博弈里的子profile
        subgame_profile = tuple(int(profile[int(p) - 1]) for p in players)
        # 取出该game对应的payoff矩阵
        subgame_payoffs = payoffs[game_name]
        # 计算这个子profile中每个玩家的收益
        payoff_for_players = get_subgame_payoff(subgame_payoffs, subgame_profile, players, num_strategies)
        # 累加到全局
        """
        i：在子博弈的局部玩家编号
        p：全局玩家编号（字符串）
        """
        for i, p in enumerate(players):
            player_payoffs[int(p) - 1] += payoff_for_players[i]
    return player_payoffs

def find_next_best_profile(profile, games, payoffs, num_strategies):
    """
    函数功能描述：
    对每一个全局局势使用MBRA策略

    参数说明：
    :param param1: 参数1的描述，包括类型和用途。
    :param param2: 参数2的描述，包括类型和用途。
    ...
    :param paramN: 参数N的描述，包括类型和用途。

    返回值说明：
    :return: 返回值的描述，包括类型和含义。
        一个元组如(1,2)表示一个共有两个玩家的博弈的全局局势
    """



    """
    对于一个局势，计算下一步最优局势
    这里的profile是一个元组
    """
    current_payoffs = calculate_global_payoffs(profile, games, payoffs, num_strategies)
    new_profile = list(profile)
    
    for player in range(len(num_strategies)):
        best_strategy = profile[player]
        best_payoff = current_payoffs[player]
        
        # 尝试改变该玩家的策略
        for s in range(1, num_strategies[player] + 1):
            if s == profile[player]:
                continue  # 当前策略
            # 构造新的局势
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


# 这里的num_startegies是形如[2,2,3]的列表，用来表示每一个玩家的策略数是多少
def get_transition_matrix(games, payoffs, num_strategies):
    """
    函数功能：
        构造同步更新下的状态转移矩阵
    
    参数：
    :param games: dict
    :param payoffs: dict
    :param num_strategies: list，每个玩家所拥有的策略数量
    
    返回：
    :return: list，下一步转移到的局势编号索引
    """


    # all_profiles数据形如[(1,1),(1,2),(2,1),(2,2)]
    all_profiles = enumerate_profiles(num_strategies)
    profile_to_index = { profile: idx+1 for idx, profile in enumerate(all_profiles) }  # 1-based
    next_profile_list = []

    for profile in all_profiles:
        # next_profile是一个元组
        next_profile = find_next_best_profile(profile, games, payoffs, num_strategies)
        index_of_next = profile_to_index[next_profile]
        next_profile_list.append(index_of_next)

    return next_profile_list




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
        'game1': 
                [
            [1, 1, 1],       # CCC
            [0.33, 0.33, 1.33], # CCN
            [0.33, 1.33, 0.33], # CNC
            [1.33, 0.33, 0.33], # NCC
            [-0.33, 0.67, 0.67],# CNN
            [0.67, -0.33, 0.67],# NCN
            [0.67, 0.67, -0.33],# NNC
            [0, 0, 0]           # NNN
        ],
        'game2': [
            [4, 4],     # 11
            [4, 0],     # 12
            [0, 4],     # 21
            [10, 10]      # 22
        ]
        # 'game3': [
        #     [4, 4],     # 11
        #     [4, 0],     # 12
        #     [0, 4],     # 21
        #     [10, 10]      # 22
        # ],
        # 'game4': [
        #     [4, 4],     # 11
        #     [4, 0],     # 12
        #     [0, 4],     # 21
        #     [10, 10]      # 22
        # ]
    }
    num_strategies = [2,2,2,2]

    TransitionMatrix = get_transition_matrix(games, payoffs, num_strategies)
    # print(TransitionMatrix)
    # find_all_cycles(TransitionMatrix)

    temp= []
    temp.append(TransitionMatrix)
    TransitionMatrix = temp
    # draw_directed_graph(TransitionMatrix)

    print(TransitionMatrix)

    # find_all_cycles(TransitionMatrix)

    # predecessors = find_predecessors([16],TransitionMatrix)
    # print(predecessors)
    # print(find_max_mus([16], predecessors, num_strategies))



"""
下面是两人和三人的公共物品博弈收益矩阵，每人最多贡献1，乘数r=2
    令1=C(cooperation),2=N
game1->1,game2->2,game3->1
    
1.
[
    [1, 1, 1],       # CCC
    [0.33, 0.33, 1.33], # CCN
    [0.33, 1.33, 0.33], # CNC
    [1.33, 0.33, 0.33], # NCC
    [-0.33, 0.67, 0.67],# CNN
    [0.67, -0.33, 0.67],# NCN
    [0.67, 0.67, -0.33],# NNC
    [0, 0, 0]           # NNN
]

2.
[
    [1, 1],     # CC
    [0, 1],     # CN
    [1, 0],     # NC
    [0, 0]      # NN
]

[28, 32, 32, 28, 28, 32, 32, 28, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 28, 32, 32, 28, 28, 32, 32, 28, 28, 32, 32, 28, 28, 32, 32, 28, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 28, 32, 32, 28, 28, 32, 32, 28]


"""


"""
其他不变，只是修改双人博弈为


[
    [1, 1],     # CC
    [0, 1],     # CN
    [1, 0],     # NC
    [4, 4]      # NN
]

[28, 32, 32, 28, 28, 32, 32, 28, 64, 64, 64, 64, 64, 64, 64, 64, 60, 64, 64, 60, 60, 64, 64, 60, 32, 32, 32, 32, 32, 32, 32, 32, 28, 32, 32, 28, 28, 32, 32, 28, 64, 64, 64, 64, 64, 64, 64, 64, 60, 64, 64, 60, 60, 64, 64, 60, 32, 32, 32, 32, 32, 32, 32, 32]


"""





"""
[52, 52, 52, 52, 60, 60, 60, 60, 56, 56, 56, 56, 64, 64, 64, 64, 52, 52, 52, 52, 60, 60, 60, 60, 56, 56, 56, 56, 64, 64, 64, 64, 52, 52, 52, 52, 60, 60, 60, 60, 56, 56, 56, 56, 64, 64, 64, 64, 52, 52, 52, 52, 60, 60, 60, 60, 56, 56, 56, 56, 64, 64, 64, 64]
所有环：
[52]
[60, 56]
[64]

games = {
    'game1': ['1', '2', '3'],
    'game2': ['3', '4'],
    'game3': ['4', '5', '6'],
    }
    
    # 这里 payoffs 你需要补充真实数值
    payoffs = {
        'game1': 
                [
            [1, 1, 1],       # CCC
            [0.33, 0.33, 1.33], # CCN
            [0.33, 8, 0.33], # CNC
            [1.33, 0.33, 0.33], # NCC
            [3, 0.67, 0.67],# CNN
            [0.67, -0.33, 0.67],# NCN
            [0.67, 6, -0.33],# NNC
            [6, 8, 3]           # NNN
        ],
        'game2': [
            [4, 4],     # 11
            [4, 0],     # 12
            [0, 4],     # 21
            [10, 10]      # 22
        ],
        'game3':[
            [1, 1, 1],       # CCC
            [0.33, 0.33, 1.33], # CCN
            [0.33, 1.33, 0.33], # CNC
            [1.33, 0.33, 0.33], # NCC
            [-0.33, 0.67, 0.67],# CNN
            [0.67, -0.33, 0.67],# NCN
            [0.67, 0.67, -0.33],# NNC
            [0, 0, 0]           # NNN
        ]
    }
"""



"""
[5, 7, 14, 16, 13, 15, 6, 8, 5, 7, 14, 16, 13, 15, 6, 8]
所有环：
[13]
[6, 15]
[8]

num_strategies = [2,2,2,2]

games = {
    'game1': ['1', '2', '3'],
    'game2': ['3', '4']
    }
    
    # 这里 payoffs 你需要补充真实数值
    payoffs = {
        'game1': 
                [
            [1, 1, 1],       # CCC
            [0.33, 0.33, 1.33], # CCN
            [0.33, 1.33, 0.33], # CNC
            [1.33, 0.33, 0.33], # NCC
            [-0.33, 0.67, 0.67],# CNN
            [0.67, -0.33, 0.67],# NCN
            [0.67, 0.67, -0.33],# NNC
            [0, 0, 0]           # NNN
        ],
        'game2': [
            [4, 4],     # 11
            [4, 0],     # 12
            [0, 4],     # 21
            [10, 10]      # 22
        ]
    }


"""