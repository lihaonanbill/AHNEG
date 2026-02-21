# parameter: hypergraph of game

import hypernetx as hnx
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx  # 用于生成节点布局
from matplotlib.patches import Patch  # 用于图例


# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10  # 控制字体大小

# 游戏参与关系
games = {
    'game1': ['1', '2', '3', '4', '5'],
    'game2': ['1', '6', '7'],
    'game3': ['3', '8', '9'],
    'game4': ['4', '10', '11'],
    'game5': ['5', '12', '13', '14']
    }

# 创建超图
H = hnx.Hypergraph(games)

# 提取普通图（双侧图）中的节点用于生成 spring layout
# 方法：构建 bipartite network（节点+超边）
G = H.bipartite()  # 返回 networkx.Graph 对象
pos_all = nx.spring_layout(G, seed=1)  # 固定 seed 保证每次一样

# 从 bipartite 图中提取节点位置（只保留节点的部分）
pos = {n: pos_all[n] for n in H.nodes}  # 只用原始节点位置

# # 所有节点设为黑色、大小一致（如 200）
# num_nodes = len(H.nodes)
# node_colors = ['black'] * num_nodes
# node_sizes = [200] * num_nodes


# 设置节点颜色：1,2,4 -> 7BD144 (绿色)，3,5,6 -> E74F4C (红色)
# color_map = {
#     '1': '#7BD144',
#     '2': '#7BD144',
#     '4': '#7BD144',
#     '3': '#E74F4C',
#     '5': '#E74F4C',
#     '6': '#E74F4C'
# }

# node_colors = [color_map.get(node, 'black') for node in H.nodes]



# 绘图
plt.figure(figsize=(5, 4))  # 紧凑图像
hnx.drawing.draw(
    H,
    with_edge_labels=True,
    pos=pos,
    node_radius=2,
    # nodes_kwargs={
    #     'color': node_colors
    # },
    edges_kwargs={
        'linewidth': 1.2
    }
)


# 添加图例
# legend_elements = [
#     Patch(facecolor='#7BD144', edgecolor='k', label='update'),
#     Patch(facecolor='#E74F4C', edgecolor='k', label='retain')
# ]
# plt.legend(handles=legend_elements, loc='best')



plt.tight_layout()
plt.show()

