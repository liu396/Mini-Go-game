import sys
import random
import timeit
import math
import argparse
from collections import Counter
from copy import deepcopy

BOARD_SIZE = 5

ONGOING=-1
DRAW=0
BLACK_WIN=1
WHITE=WIN=2


class GoBoard:

    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 # Trace the number of moves
        self.max_move = n * n - 1 # The max movement of a Go game
        self.komi = n/2 # Komi rule
        self.verbose = True # Verbose only when there is a manual player
        self.result = 0

    def init_board(self):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(self.size)] for y in range(self.size)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.n_move = 0
        self.result = 0
        self.board = board
        self.previous_board = deepcopy(board)

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def encode_state(self):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(self.board[i][j]) for i in range(self.size) for j in range(self.size)])

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_diagonal(self,i,j):
        diagonals = []
        if i>0 and j>0:
            diagonals.append((i-1,j-1))
        if i>0 and j<self.size-1:
            diagonals.append((i-1,j+1))
        if i<self.size-1 and j>0:
            diagonals.append((i+1,j-1))
        if i<self.size-1 and j<self.size-1:
            diagonals.append((i+1,j+1))
        return diagonals

    def is_eye(self,i, j, piece_type):
        interesting_pos=self.detect_neighbor(i,j)+self.detect_diagonal(i,j)
        for pos in interesting_pos:
            if self.board[pos[0]][pos[1]]!= piece_type:
                return False
        return True


    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''

        # print("Intended placement: ",piece_type,i,j)
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            self.n_move = self.max_move+1
            self.fill_board(3-piece_type)
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        self.died_pieces = self.remove_died_pieces(3 - piece_type)
        # Remove the following line for HW2 CS561 S2020
        self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True

    def fill_board(self,piece_type):
        board=self.board
        for i in range(self.size):
            for j in range(self.size):
                board[i][j] = piece_type
        self.update_board(board)

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board

    def visualize_board(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)

    def game_end(self, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def weighted_score(self,piece_type):
        weight = [-0.2,0.1,0.5,0.8,1.0,1.1,1.2]
        visited = set()
        factor = 1
        if piece_type == 2:
            factor = 1.5
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type and (i, j) not in visited:
                    allies = self.ally_dfs(i,j)
                    qi = self.find_liberty(i,j)
                    for stone in allies:
                        if qi > 7: qi = 7
                        cnt += factor * weight[qi-1]+1
                        visited.add(stone)
        return cnt

    def advanced_score(self, piece_type):
        visited=set()
        eyes_info = {}
        board = self.board
        cnt = 0
        factor = 1
        if piece_type == 2:
            factor = 1.0

        for i in range(self.size):
            for j in range(self.size):
                if board[i][j]==0 and self.is_eye(i,j,piece_type):
                    stone = self.detect_neighbor(i,j)[0]
                    allies = self.ally_dfs(stone[0],stone[1])
                    for single in allies:
                        if single not in eyes_info:
                            eyes_info[single] = 1
                        else:
                            eyes_info[single] += 1
        for element in eyes_info:
            if eyes_info[element] > 1:
                cnt += factor * 1.5 + 1
                visited.add(element)

        weight = [-0.2,0.1,0.5,0.8,1.0,1.1,1.2]
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type and (i, j) not in visited:
                    allies = self.ally_dfs(i,j)
                    qi = self.find_liberty(i,j)
                    for stone in allies:
                        if qi > 7: qi = 7
                        cnt += factor * weight[qi-1]+1
                        visited.add(stone)
        return cnt

    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        # print("1 score: ",cnt_1)
        # print("2 score: ", cnt_2)
        # print("komi: ",self.komi)
        if cnt_1 > cnt_2 + self.komi: self.result = 1
        elif cnt_1 < cnt_2 + self.komi: self.result = 2

    # following methods helps MCTS

    def __hash__(self):
        key = self.encode_state()+' '+str(self.n_move)
        return hash(key)

    def __eq__(self, other):
        if self.n_move == other.n_move and self.board == other.board:
            return True
        else:
            return False










