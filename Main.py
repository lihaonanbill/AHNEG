from Asynchronous import *
from MatrixOperation import *

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

    # TransitionMatrix = get_state_driven_transition_matrix(
    #     games, payoffs, num_strategies, 
    #     [1,1,1,1,
    #      2,1,1,1,
    #      1,8,16,4,
    #      2,1,1,6
    #      ]
    #     )
    # print(TransitionMatrix)
    

    TransitionMatrix = get_asynchronous_transition_matrices(
        games, payoffs, num_strategies, 
        [[1,1,1,1,1],
         [2,1,1,1,1],
         [1,2,1,1,1],
         [2,2,1,1,1]
         
         ]
        
        
        )
    

    TPM_action = [
        [0.1,0.4,0.2,0.3],
        [0.2,0.1,0.1,0.1],
        [0.3,0.2,0.4,0.2],
        [0.4,0.3,0.3,0.4]
    ]
    


    # --模型3
    print("TransitionMatrix",TransitionMatrix)

    TPM_aug = get_augmented_matrix(TPM_action, TransitionMatrix)

    print("TPM_aug", TPM_aug)
    print("TPM_aug.shape", TPM_aug.shape)
    np.savetxt("matrix_output.txt", TPM_aug, fmt="%.4f", delimiter=", ")
    print(f"矩阵已保存至 {"matrix_output.txt"}")

    result = check_close_set_aug(32, 4, 32, TPM_aug, 2)
    print(result)

    # find_all_cycles(TransitionMatrix)

    # temp= []
    # temp.append(TransitionMatrix)
    # TransitionMatrix = temp

    

    # draw_directed_graph(TransitionMatrix)
    # draw_directed_graph_with_beauty(TransitionMatrix)

    # --模型 2

    # print(weighted_sum_logical_matrices(TransitionMatrix, [0.25,0.25,0.25,0.25]))
    # predecessors = find_predecessors([32],TransitionMatrix)
    # print("前驱节点(能达到的节点)",predecessors)
    # print(find_max_mus([32], predecessors, num_strategies))

    # print(check_close_set([32], 2, num_strategies, TransitionMatrix, predecessors))

    # --模型2的全概率版本

    probability_update = [0.5,0.7,0.3,0.8,0.4]

    asynchronous_schemes, weights = get_weights_and_all_asynchronous_schemes(probability_update)


    weighted_matrices = weighted_sum_logical_matrices(
        get_asynchronous_transition_matrices(games, payoffs, num_strategies, asynchronous_schemes),
        weights
        )
    
    print(weighted_matrices)
    # 保存到文件中
    np.savetxt("weighted_matrices.txt", weighted_matrices, fmt="%.4f", delimiter=", ")
    print(f"矩阵已保存至 {"weighted_matrices.txt"}")


    






"""
state_driven_asynchronous

"""
