# Project Description: Construction of State Transition Matrix for Hypernetwork Evolutionary Game

## Module Call Relationship
    Game <- StateTransitionGraph
    Asynchronous <- Game
    Main <- Asynchronous, MatrixOperation

## Module Functions:
    Main: 
        The main module where all modules are centrally tested
    MatrixOperation:
        1. Implementation of the semi-tensor product of matrices
        2. Calculation of the augmented matrix (Model 3)
    Game:
        Calculation of the transition matrix under synchronous conditions
    Asynchronous:
        Calculation of the transition matrix under asynchronous conditions
    StateTransitionGraph:
        Perform some verifications using graph methods
        1. Identify cycles in the graph (mainly used to find fixed points under synchronous conditions, as fixed points in synchronous conditions are necessarily fixed points in asynchronous conditions and also potential evolutionarily stable patterns in asynchronous conditions)
        2. Find the basin of attraction for a specific state
        3. Determine the potential maximum mu value of a specific fixed point
        4. Judge whether it is a mu-ESP