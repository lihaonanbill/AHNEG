from itertools import product
import networkx as nx
from scipy.linalg import khatri_rao
from math import lcm
import numpy as np


# 计算多个逻辑矩阵的加权和
def weighted_sum_logical_matrices(logical_matrices, weights):
    """
    对多个逻辑矩阵（用列表形式表示）按权重加权求和。
    
    参数：
        logical_matrices: List[List[int]]
            每个内部列表表示一个逻辑矩阵的列索引位置（1-based），
            每列为一个 one-hot 向量。
        weights: List[float]
            与每个逻辑矩阵对应的权重，长度应与 logical_matrices 相同。
            
    返回：
        np.ndarray
            加权求和后的矩阵（每列为浮点值）
    """
    assert len(logical_matrices) == len(weights), "矩阵数量与权重数量不一致"
    n_rows = max(max(mat) for mat in logical_matrices)  # 行数为最大索引值
    n_cols = len(logical_matrices[0])  # 列数等于每个逻辑矩阵的长度（应一致）

    result = np.zeros((n_rows, n_cols))

    for mat, w in zip(logical_matrices, weights):
        temp = np.zeros((n_rows, n_cols))
        for j, row_index in enumerate(mat):
            temp[row_index - 1, j] = 1  # 转换为 0-based 索引
        result += w * temp

    return result

# 实现两个矩阵之间的半张量积
def STP(A, B):
    """
    Semi-Tensor Product of matrices A and B.

    Parameters:
        A : np.ndarray of shape (m, x)
        B : np.ndarray of shape (y, t)

    Returns:
        np.ndarray of shape (m * (β // x), t * (β // y))
    """
    m, x = A.shape
    y, t = B.shape
    beta = lcm(x, y)

    I_beta_x = np.eye(beta // x)
    I_beta_y = np.eye(beta // y)

    A_kron = np.kron(A, I_beta_x)
    B_kron = np.kron(B, I_beta_y)

    return A_kron @ B_kron
    


# 在一部动作符合一个齐次马尔可夫链的情况下将一部动作和博弈局势进行耦合增广，由于矩阵位数相同，半张量积可以直接用普通矩阵乘法替代
def get_augmented_matrix(TPM_action, edge_lists):
    """
    函数功能描述：
        1.首先进行矩阵转换
            利用dim_action获得单位矩阵I_{dim_action}
            利用dim_profile获得全为1的行向量1^{T}_{dim_profile}
            利用edge_list获得一个dim_profile x (dim_action x dim_profile)大小的逻辑矩阵L(每一列只有一个元素是1，其余全为0，这个1所在行号就是edge_lists中每个列表中的每个数字元素)
        2.进行乘法操作
            TPM_aug = (TPM_action x (I_{dim_action} · 1^{T}_{dim_profile}))*L
            其中x表示矩阵的半张量积， ·表示Kronecker product, *表示Khatri-Rao product
        
    
    参数：
    :param TPM_action： 异步动作的转移概率矩阵
    :param dim_action：异步动作\theta的维数
    :param dim_profile：博弈局势的维数
    :param edge_lists：包含多个子网络的转移矩阵
    
    返回：
    :return: TPM_aug
    """

    dim_action = len(edge_lists)
    dim_profile = len(edge_lists[0])

    # Step 1: 构造 I ⊗ 1^T
    I_theta = np.eye(dim_action)
    one_T_profile = np.ones((1, dim_profile))
    kron_block = np.kron(I_theta, one_T_profile)  # shape: (dim_action, dim_action * dim_profile)

    # Step 2: 半张量积
    TPM_action = np.array(TPM_action)
    stp_result = STP(TPM_action, kron_block)  # shape: (dim_action, dim_action * dim_profile)

    # Step 3: 构造逻辑矩阵 L
    n_cols = dim_action * dim_profile
    L = np.zeros((dim_profile, n_cols))
    for a_idx, edges in enumerate(edge_lists):
        for p_idx, target in enumerate(edges):
            col_idx = a_idx * dim_profile + p_idx
            row_idx = target - 1  # 转为0-based索引
            L[row_idx, col_idx] = 1


    # Step 4: Khatri-Rao 乘法（列对应）
    # print("stp_result",stp_result.shape)
    # print("L",L.shape)

    TPM_aug = khatri_rao(stp_result, L)  

    return TPM_aug


if __name__ == "__main__":
    # X = [
    # [2, -2, -1, 1],
    # [1, 0, 3, -3],
    # [-2, -3, 2, 1]
    # ]
    # Y = [
    # [-2, 1],
    # [-3, 2]
    # ]
    # A=np.array(X)
    # B=np.array(Y)
    # print(STP(A,B))


    logical_matrices = [
        [1,1,3,4],
        [1,3,2,1]
    ]

    weights = [
        0.5, 0.5    
    ]
    print(weighted_sum_logical_matrices(logical_matrices, weights))