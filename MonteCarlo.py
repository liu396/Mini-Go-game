import math
from copy import deepcopy
import random
from collections import defaultdict
from GoBoard import GoBoard
import numpy as np

def decode_state(code,size):
    board = [[0 for x in range(size)] for y in range(size)]
    for x in range(size):
        for y in range(size):
            board[x][y] = code[x*size+y]
    return board

def find_possible(board, side):
    '''find the possible move of current board'''
    possible_placements = []
    for i in range(board.size):
        for j in range(board.size):
            if board.valid_place_check(i, j, side, test_check=True):
                possible_placements.append((i, j))
    return possible_placements


def find_random_child(board):
    '''find a random successive state after a random move'''
    temp = deepcopy(board)
    if board.n_move%2 == 0:
        piece_type = 1
    else:
        piece_type = 2
    possible_pos = find_possible(temp,piece_type)
    if not possible_pos:
        temp.n_move = temp.max_move+1
        temp.fill_board(3-piece_type)
        return temp

    placement = random.choice(possible_pos)
    temp.place_chess(placement[0],placement[1],piece_type)
    return temp


def find_children(board):
    '''find all possible successive states'''
    children = set()
    if board.game_end():
        return children
    temp = deepcopy(board)
    if board.n_move%2 == 0:
        piece_type = 1
    else:
        piece_type = 2
    possible_pos = find_possible(temp,piece_type)
    if not possible_pos:
        temp = deepcopy(board)
        temp.place_chess(-1, -1, piece_type)
        return children.add(temp)
    for pos in possible_pos:
        temp = deepcopy(board)
        temp.place_chess(pos[0],pos[1],piece_type)
        children.add(temp)
    return children


class MCTS:
    "Monte Carlo Tree Search"

    def __init__(self, c=1, init_side=None, max_exploration=100, k=0):
        self.c = c
        self.Q = defaultdict(int)  # reward of each node #
        self.N = defaultdict(int)  # Total visit of each node #
        self.children = {}  # All nodes of the tree #
        self.init_side = init_side
        self.num_comparison = 0
        self.max_exploration = max_exploration
        self.k = k # want to set a threshold for expansion, i.e. leaf get expanded only it is visted more than k times. Has some problems if k is greater than 0. Need to be fix.

    def set_side(self,side):
        self.init_side = side

    def choose(self,node):
        '''choose the best move'''
        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            # print("n info: ")
            # print(n.board)
            # print(n.n_move)
            return self.Q[n] / self.N[n]  # average reward

        if node not in self.children:
            print("Not a visited state\n")
            target = find_random_child(node)
        else:
            target = max(self.children[node], key=score)
        # print("Node: ",node.board)
        # print(node.n_move)
        # print("Children: ")
        # for element in self.children[node]:
            # print(element.board)
            # print(element.n_move)
        print("Next move information: ", self.Q[target], self.N[target])
        # print("target: ",target.board)
        # print(target.n_move)


        possible_pos = find_possible(node, self.init_side)
        # print(possible_pos)
        for pos in possible_pos:
            temp = deepcopy(node)
            temp.place_chess(pos[0],pos[1],self.init_side)
            # print("Possible move value: ", self.Q[temp], self.N[temp])
            # print("pos:",pos[0],pos[1])
            # print("temp: ",temp.board)
            if temp.board == target.board:
                move =(pos[0], pos[1])
        return move

    def rollout(self,node):
        '''do one simulation and get one more node of the tree'''
        self.num_comparison = 0
        path = self.select(node)
        # print("Number of explored depth: ",len(path))
        # print("Num_of_comparison of uct values: ",self.num_comparison)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self,node):
        '''select a path that get to a leaf node based on uct values '''
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path

            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.uct_select(node)

    def uct_select(self,node):
        '''balancing exploration and exploitation'''
        # assert all (n in self.children for n in self.children[node])
        # print(self.N[node])
        lnt = math.log(self.N[node])

        def uct(n):
            value = self.Q[n] / self.N[n] + self.c * math.sqrt(lnt / self.N[n])
            # print("UCT information: ",self.Q[n],self.N[n],self.N[node])
            # print("uct values: ",value)
            return value

        self.num_comparison += len(self.children[node])
        return max(self.children[node], key=uct)

    def expand(self, node):
        "Update the `children` dict with the children of `node`"
        # if node.game_end() or node.n_move >= self.max_exploration:
        #     self.children[node] = set()
        #     return
        if node in self.children:
            print("Found following: ")
            print(node.board)
            return  # already expanded
        if self.N[node]<self.k:
            return
        self.children[node] = find_children(node)
        # if self.children[node] and node.n_move != self.children[node].pop().n_move-1:
        #     raise RuntimeError("Wrong parents!")
        return

    def simulate(self,node):
        while True:
            if node.game_end():
                node.judge_winner()
                if node.result == self.init_side:
                    reward = 1
                elif node.result == 3 - self.init_side:
                    reward = -1
                else:
                    reward = 0
                if node.n_move % 2 == 0 and self.init_side == 1:
                    reward = - reward
                return reward
            node = find_random_child(node)

    def backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = - reward # check if this is correct. reward = 1- reward if win is 1 draw is 0.5 and lose is zero.



class MCTSGo:
    def __init__(self,max_iteration=200,side=None, Load=False,clean_tree=False):
        self.side = side
        self.max_iter = max_iteration
        self.verbose = False
        self.clean_tree=clean_tree # get rid of learned tree to get faster move. Bad for learning
        if Load:
            # Load from table. Need to be completed
            self.tree=MCTS()

            f = open('mcts_tree.txt','r')
            lines = f.readlines()
            size = int(math.sqrt(len(lines[0])))
            total = int(len(lines)/4)
            for i in range(total):
                node = GoBoard(size)
                node.update_board(decode_state(lines[total*4],size))
                step,q,n = [int(w) for w in lines[total*4+1].strip().split(' ')]
                node.n_move = step
                repeat = int(lines[total*4+2])
                if repeat == 0:
                    continue

                self.tree.Q[node] = q
                self.tree.N[node] = n
                self.tree.children[node] = set()

                sons = [str(n) for n in lines[total*4+3].strip().split(' ')]
                for j in repeat:
                    child = GoBoard(size)
                    child.update_board(decode_state(sons[j*2],size))
                    child.n_move = int(sons[j*2+1])
                    self.tree.children[node].add(child)
        else:
            self.tree = MCTS()

    def set_side(self,side):
        self.side = side
        self.tree.set_side(side)

    def move(self,board):
        if self.clean_tree:
            self.tree = MCTS()
            self.tree.set_side(self.side)
        for i in range(self.max_iter):
            if self.verbose:
                print("iteration: ",i)
            self.tree.rollout(board)
        placement = self.tree.choose(board)
        return board.place_chess(placement[0], placement[1], self.side)

    def learn(self,board):
        return

    def write_cur_tree(self):
        # Save learned data
        f = open('mcts_tree.txt','w')
        # print(len(self.tree.children))
        for node in self.tree.children:
            if node not in self.tree.children:
                continue
            # print(node in self.tree.children)
            f.write(node.encode_state()+'\n')
            f.write(str(node.n_move)+' ')
            f.write(str(self.tree.Q[node])+' ')
            f.write(str(self.tree.N[node])+'\n')
            # print(node in self.tree.children)
            f.write(str(len(self.tree.children[node]))+'\n')
            for child_node in self.tree.children[node]:
                f.write(child_node.encode_state()+' '+str(child_node.n_move)+' ')
            f.write('\n')

        f.write('End')
        f.close()

class MCTS_ABGo:
    '''Combine MCTS and Alpha-Beta'''
    def __init__(self,max_iteration=200,side=None, Load=False, max_depth=6, max_branching=20,clean_tree=False):
        self.side = side
        self.max_iter = max_iteration
        self.verbose = False
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.clean_tree = clean_tree
        self.cur_step = 0
        self.cur_depth = 0

        if Load:
            self.tree=MCTS()

            f = open('mcts_tree.txt','r')
            lines = f.readlines()
            size = int(math.sqrt(len(lines[0])))
            total = int(len(lines)/4)
            for i in range(total):
                node = GoBoard(size)
                node.update_board(decode_state(lines[total*4],size))
                step,q,n = [int(w) for w in lines[total*4+1].strip().split(' ')]
                node.n_move = step
                repeat = int(lines[total*4+2])
                if repeat == 0:
                    continue

                self.tree.Q[node] = q
                self.tree.N[node] = n
                self.tree.children[node] = set()

                sons = [str(n) for n in lines[total*4+3].strip().split(' ')]
                for j in repeat:
                    child = GoBoard(size)
                    child.update_board(decode_state(sons[j*2],size))
                    child.n_move = int(sons[j*2+1])
                    self.tree.children[node].add(child)
        else:
            self.tree = MCTS(max_exploration=100)

    def set_side(self,side):
        self.side = side
        self.tree.set_side(side)

    def move(self,board):
        '''First steps use MCTS, last moves use AB prunning'''
        if self.clean_tree:
            self.tree=MCTS(max_exploration=100)
            self.tree.set_side(self.side)

        if board.n_move >= board.max_move-6:
            print("AB MOVE\n")
            place = self.AB_search(board)
            if place[0] > 100:
                # if AB search tells that you are gonna lose anyway, use MCTS, hoping opponent makes some mistakes.
                print("Take Chances")
            else:
                return board.place_chess(place[0], place[1], self.side)

        print("MC MOVE\n")
        for i in range(self.max_iter):
            if self.verbose:
                print("iteration: ",i)
            self.tree.rollout(board)
        placement = self.tree.choose(board)
        return board.place_chess(placement[0], placement[1], self.side)

    def learn(self,board):
        return

    def AB_search(self, board):
        self.cur_step = board.n_move
        self.cur_depth = 0
        temp_board = deepcopy(board)
        v, place = self.max_value(temp_board, -np.inf, np.inf)
        print("Best Assured Value: ",v)
        if v <= 0:
            # Will lose if opponent behave perfectly. Return an invalid move. The "move" method will handle this.
            place = (500,500)
        return place

    def max_value(self,board, alpha, beta):
        # print("Max_value called: ")
        # print("Current depth: ", self.cur_depth)
        # print("Current step: ",self.cur_step)

        if self.cur_step>=board.max_move or board.game_end():
            # print("End value, max, absolute score\n")
            # return board.score(self.side) - board.score(3 - self.side), (-1, -1)
            board.judge_winner()
            return int(board.result == self.side), (-1,-1)

        if self.cur_depth >= self.max_depth:
            # print("End value, max, weighted score\n")
            return board.weighted_score(self.side) - board.weighted_score(3 - self.side), (-1, -1)

        possible_pos = find_possible(board,self.side)
        # print("Possible Pos: ",possible_pos)
        if not possible_pos:
            print("ABGo no where to go")
            return -board.score(3-self.side),(-1,-1)

        if len(possible_pos) > self.max_branching:
            random.shuffle(possible_pos)
            possible_pos = possible_pos[:self.max_branching]

        v = -np.inf
        max_pos = (-1,-1)
        for pos in possible_pos:
            temp_board = deepcopy(board)
            self.cur_depth += 1
            self.cur_step += 1
            temp_board.place_chess(pos[0],pos[1],self.side)
            v = max(v, self.min_value(temp_board, alpha, beta))
            self.cur_depth -= 1
            self.cur_step -= 1
            if v >= beta:
                del temp_board
                return v, max_pos
            if v > alpha:
                max_pos = pos
                alpha = v

        del temp_board
        return v, max_pos

    def min_value(self,board,alpha,beta):
        # print("Min_value called")
        # print("Current depth: ", self.cur_depth)
        # print("Current step: ",self.cur_step)

        if self.cur_step >= board.max_move or board.game_end():
            print("End value, min, absolute score\n")
            return board.score(self.side) - board.score(3 - self.side)

        if self.cur_depth >= self.max_depth:
            print("End value, min, weighted score\n")
            return board.weighted_score(self.side) - board.weighted_score(3 - self.side)

        possible_pos = find_possible(board, 3-self.side)
        # print("Possible Pos: ", possible_pos)
        if not possible_pos:
            return board.score(self.side)

        if len(possible_pos) > self.max_branching:
            random.shuffle(possible_pos)
            possible_pos = possible_pos[:self.max_branching]

        v = np.inf
        for pos in possible_pos:
            temp_board = deepcopy(board)
            self.cur_depth += 1
            self.cur_step += 1
            temp_board.place_chess(pos[0],pos[1],3-self.side)
            v_2, dummy = self.max_value(temp_board, alpha, beta)
            self.cur_depth -= 1
            self.cur_step -= 1
            v = min(v, v_2)
            if v <= alpha:
                del temp_board
                return v
            beta = min(beta,v)

        return v





