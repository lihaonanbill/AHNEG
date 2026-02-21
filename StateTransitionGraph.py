import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from collections import deque
import math
import numpy as np



def draw_directed_graph_with_beauty(edge_lists):
    """
    功能：
        将多个转移矩阵合并成一个图，绘制有向图：
        - 不动点（自环）放圆心
        - 其他节点按最短距离放同心圆
        - 边为浅灰色
        - 不动点红色，其他节点蓝色
    """

    G = nx.DiGraph()

    # 合并图
    n_nodes = max(len(edge) for edge in edge_lists)
    for i in range(1, n_nodes + 1):
        G.add_node(i)

    for edge_list in edge_lists:
        for i, target in enumerate(edge_list):
            G.add_edge(i+1, target)

    # 寻找不动点
    fixed_points = [
        i for i in range(1, n_nodes+1)
        if G.has_edge(i, i) and G.out_degree(i) == 1
    ]

    # 计算每个节点到最近不动点的最短距离
    distances = {}
    for node in G.nodes:
        if node in fixed_points:
            distances[node] = 0
        else:
            min_dist = math.inf
            for fp in fixed_points:
                try:
                    d = nx.shortest_path_length(G, node, fp)
                    if d < min_dist:
                        min_dist = d
                except nx.NetworkXNoPath:
                    continue
            if min_dist < math.inf:
                distances[node] = min_dist
            else:
                distances[node] = -1  # 不可达

    # 按照同心圆布置
    # 每层圆的半径
    r_step = 2
    pos = {}
    # 先把圆心位置固定
    center_x, center_y = 0, 0
    for idx, fp in enumerate(fixed_points):
        # 如果多个不动点，就均匀分布在一个很小的圆里
        angle = idx * 2 * math.pi / len(fixed_points)
        pos[fp] = (center_x + 0.1 * math.cos(angle), center_y + 0.1 * math.sin(angle))
    # 其他节点
    max_layer = max(distances.values())
    for layer in range(1, max_layer+1):
        layer_nodes = [node for node, d in distances.items() if d==layer]
        n_layer_nodes = len(layer_nodes)
        for idx, node in enumerate(layer_nodes):
            angle = idx * 2 * math.pi / n_layer_nodes
            r = r_step * layer
            pos[node] = (center_x + r * math.cos(angle), center_y + r * math.sin(angle))

    # 不可达节点放最外一圈
    unreachable_nodes = [node for node,d in distances.items() if d==-1]
    for idx, node in enumerate(unreachable_nodes):
        angle = idx * 2 * math.pi / max(len(unreachable_nodes),1)
        r = r_step * (max_layer+1)
        pos[node] = (center_x + r * math.cos(angle), center_y + r * math.sin(angle))

    # 画图
    node_colors = ['red' if n in fixed_points else 'lightblue' for n in G.nodes]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        node_color=node_colors,
        arrowsize=15,
        edge_color='lightgray',
        font_size=10
    )
    plt.title("Directed Graph with Fixed Points (Concentric Circles)")
    plt.show()




def draw_directed_graph(edge_lists):
    """
    函数功能：
        根据多个转移矩阵画出有向图
    
    参数：
    :param edge_list: list
    
    
    返回：
    :return: 
    """


    G = nx.DiGraph()

    # 先确定最大节点编号
    n_nodes = max(len(edge) for edge in edge_lists)
    for i in range(1, n_nodes + 1):
        G.add_node(i)

    # 把所有edge_list里的边都添加
    for edge_list in edge_lists:
        for i, target in enumerate(edge_list):
            G.add_edge(i + 1, target)

    pos = nx.shell_layout(G)

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1000,
        node_color='lightblue',
        arrowsize=20,
        font_size=12
    )
    plt.title("多个情景的有向边叠加可视化")
    plt.show()


def find_all_cycles(edge_lists):
    """
    功能：
        输入一个列表，列表中每一个元素都是一个转移矩阵（edge list）
        将这些转移矩阵合并成一个大图，然后寻找所有的环
    参数：
        graph_list: list of list[int]
            例如 [ [2,1,3], [3,2,1] ]
    返回：
        cycles: list of list[int]
    """
    if not edge_lists:
        print("没有输入。")
        return []

    # 假设所有graph长度一样
    n = len(edge_lists[0])
    combined_graph = [set() for _ in range(n)]

    # 合并
    for edge_list in edge_lists:
        for i, target in enumerate(edge_list):
            combined_graph[i].add(target)

    # 把集合再转成列表用于DFS搜索
    graph = [list(targets) for targets in combined_graph]

    visited = [0] * n
    path = []
    cycles = []

    def dfs(node):
        visited[node] = 1
        path.append(node + 1)

        for next_node in graph[node]:
            next_node -= 1  # 转成0下标
            if visited[next_node] == 0:
                dfs(next_node)
            elif visited[next_node] == 1:
                # 找到一个环
                cycle_start = path.index(next_node + 1)
                cycle = path[cycle_start:]
                if cycle not in cycles:
                    cycles.append(cycle)

        visited[node] = 2
        path.pop()

    for i in range(n):
        if visited[i] == 0:
            dfs(i)

    if not cycles:
        print("图中不存在环。")
    else:
        print("所有环：")
        for c in cycles:
            print(c)
    return cycles


def find_predecessors(dest_stats, edge_lists):
    """
    功能：
        找出每一个不动点的所有前驱节(吸引域),为判断其是否为一个演化稳定局势做准备
        这里用的是BFS,还有一种方法是骆超论文中提到的迭代把上一步的predecessor加入集合，知道该集合饱和

        其实骆超论文中的方法是从dest_state开始的BFS,下面应该也已经改成了从dest_state开始的BFS

    参数：
    :param dest_states: list 目标节点,是一个列表，保存了所有的目标节点
    :param edge_lists: list 表示节点间的连接状态
    :
    返回：
    :return: list,list中的每一个元素也是一个列表，
    保存了dest_states中每个状态对应的所有前驱节点(可到达该dest_state所有状态)
    
    """

    n_nodes = len(edge_lists[0])
    total_states = n_nodes

    # 构造合并后的邻接矩阵（逻辑转移矩阵），转置为方便处理“反向边”
    adj_matrix = np.zeros((total_states, total_states), dtype=int)  # A[i, j] = 1 表示 j→i

    for edge_list in edge_lists:
        for i, j in enumerate(edge_list):
            adj_matrix[j-1, i] = 1  # 注意是 j→i，表示从i到j的边，现在转置为 j←i

    predecessors = []

    for dest in dest_stats:
        visited = set([dest - 1])  # 从编号转为索引
        queue = deque([dest - 1])

        while queue:
            curr = queue.popleft()
            for i in range(total_states):
                if adj_matrix[curr, i] > 0 and i not in visited:
                    visited.add(i)
                    queue.append(i)

        # 将索引转为状态编号（从 1 开始）
        predecessors.append([v + 1 for v in visited])

    return predecessors

def enumerate_profiles(num_strategies):
    return list(product(*[range(1, s+1) for s in num_strategies]))

# 求状态相关异步中某一个不动点的最小mu值
def find_max_mus(dest_stats, predecessors, num_strategies):
    """
    功能：
        计算每个dest_state的最大mu
        (保证所有距离≤mu的状态都包含在 predecessors∪dest 中，且 mu 尽可能大)
    参数：
        dest_stats: list[int]
            目标节点编号
        predecessors: list[list[int]]
            每个目标节点的所有前驱节点
        num_strategies: list[int]
            每个玩家的策略数量
    返回：
        list[ (int, list[int]) ]
            (最大mu, 所有距离≤mu的状态编号)
    """
    profiles = enumerate_profiles(num_strategies)
    results = []

    for idx, dest in enumerate(dest_stats):
        dest_profile = profiles[dest - 1]  # 因为编号从 1 开始
        pred_set = set(predecessors[idx])

        mu = 0
        max_mu = -1
        best_close_states = []
        mu_upper_bound = len(num_strategies)

        while mu <= mu_upper_bound:
            close_states = []
            for state_idx, prof in enumerate(profiles, start=1):
                distance = sum(1 for a, b in zip(prof, dest_profile) if a != b)
                if distance <= mu:
                    close_states.append(state_idx)
            # 判断是否所有距离≤mu的状态都在前驱
            if set(close_states).issubset(pred_set.union({dest})):
                # 满足条件，继续增大
                max_mu = mu
                best_close_states = close_states
                mu += 1
            else:
                # 第一次失败就返回上一次满足条件的
                results.append( (max_mu, best_close_states) )
                break
        else:
            # while循环自然结束
            results.append( (max_mu, best_close_states) )

    return results




"""
我们来这样找一个图中的最大封闭集和，用迭代的方法，
1.首先找出图中所有节点的一步可达集，然后给出所有一步可达集的公共集合
2.将这个公共集合作为一个新集合，重复1的操作，知道公共集合不在变化
然后将上面算法加入下面算法的步骤2，
实现这样一个算法
输入有1.一个目标状态2.一个mu值3.num_strategies4.graph_lists
要求对每一个和目标值的距离为mu的状态s做这样一个判断
1.首先找出状态s的可达集
2.其次判断在这个可达集中目标状态作为一个集合是否是最大的封闭集合(不要求内部全连通，只要求这个集合封闭即可)

"""
def states_to_profiles(states, num_strategies):
    """
    输入:
        states: list[int]
            状态编号 (从 1 开始编号)
        num_strategies: list[int]
            每个玩家的策略数
    返回:
        list[tuple]
            对应的 profile
    """
    all_profiles = enumerate_profiles(num_strategies)
    profiles = []
    for s in states:
        profiles.append(all_profiles[s - 1])  # 这里 s 从 1 开始
    return profiles



def check_close_set(dest_states, mu, num_strategies, graph_lists, predecessors):
    """
    功能：
        对每个dest_state, 对所有和dest_state距离=mu的状态s:
        1) 计算s的可达集
        2) 判断可达集是否包含在dest_state的前驱节点集合中
           如果包含，就说明dest_state是该可达集的唯一封闭集

    参数：
        dest_states: list[int] 目标状态(编号从1开始)
        mu: int
        num_strategies: list[int]
        graph_lists: list of list[int]
        predecessors: list[list[int]]
            对应dest_states的前驱节点集合
    
    返回：
        dict
            key: (dest_state, s)
            value: True / False
    """

    from collections import deque
    from itertools import product
    import networkx as nx

    # 合并多个图
    n_nodes = len(graph_lists[0])
    combined_graph = [set() for _ in range(n_nodes)]
    for edge_list in graph_lists:
        for i, target in enumerate(edge_list):
            combined_graph[i].add(target)

    # 转 networkx
    G = nx.DiGraph()
    for i, targets in enumerate(combined_graph):
        for t in targets:
            G.add_edge(i+1, t)

    # 生成所有 profiles
    profiles = list(product(*[range(1, s+1) for s in num_strategies]))

    result = dict()

    for idx, dest in enumerate(dest_states):
        dest_profile = profiles[dest - 1]
        pred_set = set(predecessors[idx])

        # 先找出距离=mu的状态
        candidates = []
        for state_idx, prof in enumerate(profiles, start=1):
            dist = sum(1 for a,b in zip(prof, dest_profile) if a != b)
            if dist <= mu:
                candidates.append(state_idx)
        
        for s in candidates:
            # 计算 s 的可达集
            reachable = set()
            queue = deque([s])
            while queue:
                node = queue.popleft()
                if node not in reachable:
                    reachable.add(node)
                    for next in combined_graph[node-1]:
                        queue.append(next)

            # 判断可达集是否完全包含在dest的前驱集合(吸引域)中
            #   判断dest是否是R(s)中唯一的封闭集，判断方法是R(s)是否包含在dest的吸引域中
            #       基于这样一个proposition: dest是R(s)中唯一的封闭集当且仅当R(s)包含在dest的吸引域中
            #           充分性：若R(s)包含在。。。说明每个节点都能到达dest，那么不可能有其它封闭集(由封闭集的定义可知)
            #           必要性：若dest是R(s)中唯一封闭集，是否所有状态都能到达dest(需要证明)， (见theorem 2中的sufficiency部分)  
            """
            这里有这样一个前提，已经通过find_max_mu,去确定了predecessors中可能会收敛到不动点的状态，
            在下面这一步要确定的是不动点是否是这些候选状态可达集中唯一的封闭集合，也就是说判断上面的充分性条件即可
            """
               
            if reachable.issubset(pred_set.union({dest})):
                result[(dest, s)] = True
            else:
                result[(dest, s)] = False

    return result



def check_close_set_aug(dest_profile, dim_action, dim_profile, TPM_aug, mu):
    """
    函数功能描述：
        preliminary:
            将action和profile分别用逻辑向量\theta和e表示，逻辑向量就是在一列中只有一个元素为1，其余元素为0的列向量，比如在2维空间中，1就是[1,0]',2就是[0,1]'
            让aug表示耦合之后的逻辑向量，aug=action STP profile, 比如[1,0]'STP[0,1,0]'=[0,0,0,0,1,0]' , 也就是状态值5

            TPM_aug 相当于是关于状态aug的转移概率矩阵， 不过这个转移概率矩阵设置为列和为1而不是行和为1

        这个函数的功能就是根据Definition 10和Theorem 3中的要求来判断e^{*}是否是一个\mu-ESP，
        有一点需要说明，在Definition 10中\tilde{\theta} \in \Delta_{2^n}，而该函数中\tilde{\theta} \in dim_action
            1.首先计算出D^{*}中所有元素在增广系统中的状态值，E_{\mu}中所有状态在增广系统中的状态值
            2.然后计算将D^{*}作为一个整体，验证其是否封闭
            3.找到所有能够到达D^{*}这个集合的状态
            4.对于E_{\mu}中的每一个元素\epsilon_{0}，判断\epsilon_{0}的可达集R(\epsilon_{0})是否包含在所有能够到达D^{*}的集合中


        
        
    
    参数：
    :param dest_profile：待验证的不动点局势
    :param dim_action：异步动作\theta的维数
    :param dim_profile：博弈局势的维数
    :param TPM_aug：增广后的状态转移矩阵
    :param mu：\mu
    
    返回：
    :return: 判断dest_profile是否是mu-ESP
    """
    total_states = dim_action * dim_profile

    # Step 1: Compute D* and E_mu (indices of states in augmented system)
    D_star = [a * dim_profile + dest_profile - 1 for a in range(dim_action)]
    E_mu = []
    for a in range(dim_action):
        for e in range(dim_profile):
            dist = int(e != dest_profile - 1)
            if dist == mu:
                E_mu.append(a * dim_profile + e)

    # Step 2: Check if D* is closed: all transitions from D* must stay in D*
    for col in D_star:
        dests = np.nonzero(TPM_aug[:, col])[0]
        if not all(d in D_star for d in dests):
            return False

    # Step 3: Find all states that can reach D*
    reachable_to_D_star = set(D_star)
    queue = deque(D_star)
    while queue:
        curr = queue.popleft()
        for i in range(total_states):
            if TPM_aug[curr, i] > 0 and i not in reachable_to_D_star:
                reachable_to_D_star.add(i)
                queue.append(i)

    # Step 4: For each epsilon_0 in E_mu, check if R(epsilon_0) ⊆ reachable_to_D_star  BFS
    for eps0 in E_mu:
        visited = set()
        q = deque([eps0])
        while q:
            node = q.popleft()
            if node not in visited:
                visited.add(node)
                for i in range(total_states):
                    if TPM_aug[i, node] > 0:
                        q.append(i)
        if not visited.issubset(reachable_to_D_star):
            return False

    return True







# if __name__ == "__main__":
#     # graph = [2, 3, 4, 1]   # 节点 1->2, 2->3, 3->4, 4->1
#     graph = [28, 32, 32, 28, 28, 32, 32, 28, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 28, 32, 32, 28, 28, 32, 32, 28, 28, 32, 32, 28, 28, 32, 32, 28, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 60, 64, 64, 60, 28, 32, 32, 28, 28, 32, 32, 28]

#     find_cycle(graph)


if __name__ == "__main__":
    print(states_to_profiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16],[2,2,2,2]))