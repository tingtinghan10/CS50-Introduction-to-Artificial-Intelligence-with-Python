"""
Tic Tac Toe Player
"""

from cmath import inf
from codecs import utf_16_be_decode
from errno import ESTALE
import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    countX = 0
    countO = 0
    # rows
    for i in range(3):
        # columns
        for j in range(3):
            if board[i][j] == X:
                countX += 1
            elif board[i][j] == O:
                countO += 1
    if countX > countO:
        return O
    else:
        # X always starts first
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    # rows
    for i in range(3):
        # columns
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.update([(i, j)])
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] == EMPTY:
        newBoard = copy.deepcopy(board)
        if player(board) == X:
            newBoard[action[0]][action[1]] = X
            return newBoard
        elif player(board) == O:
            newBoard[action[0]][action[1]] = O
            return newBoard
    else:
        raise ValueError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winner_indices = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)], [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)], [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    for indices in winner_indices:
        selected = [board[index[0]][index[1]] for index in indices]
        for player in [X, O]:
            if all(item == player for item in selected):
                return player
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    noneCount = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                noneCount += 1
    if (winner(board) == None and noneCount == 0) or (winner(board) != None):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    else:
        return minimax_utility(board)[1]


def minimax_utility(board):
    if terminal(board):
        return utility(board), None

    move = None

    if player(board) == X:
        # X is the max player
        v = -math.inf
        for action in actions(board):
            util, act = minimax_utility(result(board, action))
            # if wins, return the winning move
            if util == 1:
                return util, action
            # if doesnt win, return the best move
            elif util > v:
                v = util
                move = action
        return v, move

    elif player(board) == O:
        # O is the min player
        v = math.inf
        for action in actions(board):
            util, act = minimax_utility(result(board, action))
            # if wins, return the winning move
            if util == -1:
                return util, action
            # if doesnt win, return the best move
            elif util < v:
                v = util
                move = action        
        return v, move