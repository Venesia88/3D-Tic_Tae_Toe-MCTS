from copy import deepcopy

class Node:
    def __init__(self, player, state, parent = None, move = None, wins=0, sims=0, children=[]):
        self.player = player
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = wins
        self.sims = sims
        self.children = deepcopy(children)
    
        if parent:
            parent.children.append(self)

        if move:
            self.state.update(move,player.symbol)

    def amount(self):
        return len(self.children)
    
    def make(self,move):
        return Node(self.player.opp, deepcopy(self.state), self, move)

import time
import random
import math

win = 1.0
draw = 0.5
lose = -1.0

OUTCOME_NONE = None
OUTCOME_WIN  = 1
OUTCOME_LOSE = 2
OUTCOME_DRAW = 0
now = lambda: int(round(time.time()*1000))
class Tree:
    def __init__(self, player, state, dtl = False):
        self.root = Node(
            player = player.opp,
            state = deepcopy(state),
            parent = None,
            move = None,
            wins = 0,
            sims = 0,
            children = []
        )
        # self.dtl = dtl

    def bestMove(self, dbg = False):
        most = -math.inf
        bestChild = []
        for child in self.root.children:
            wins = child.wins
            if most < wins: #choose child
                most = wins
                bestChild = [child]
            elif most == wins:
                bestChild.append(child)
        
        if len(bestChild) < 1:
            return None
        else:
            best = random.choice(bestChild)
        return best.move

    def bestChildren(self, node, C, dtl= False):
        maxC = -math.inf
        maxC_Child = []
        children = node.children

        score = []
        moves = []

        for child in children:
            c = Tree.UCT(child, C)
            if c > maxC:
                maxC = c
                maxC_Child = [child]
            elif c == maxC:
                maxC_Child.append(child)

        if len(maxC_Child) < 1:
            if node.amount():
                # return node.children[0]
                return random.choice(node.children)
        return random.choice(maxC_Child)

    @staticmethod
    def search(game, limit, constant, dbg= False, dtl = False):
        tree = Tree(game.current, deepcopy(game.board), dtl)
        interval = 0
        Tree.select(game, tree, tree.root, constant, now()+limit, now()-interval,interval,dbg,dtl)
        return tree
    
    @staticmethod
    def select(g, t, n, c, end,limit, interval, dbg,dtl):
        original = g.current

        while True:
            if now() >= end:
                return
            current = n.player.opp
            curBoard = n.state

            result = curBoard.winner()
            if result == current.symbol:
                result = 1
            elif result == -1*current.symbol:
                result = 2

            if result is not None:
                if n is t.root:
                    continue
                else:
                    Tree.backprop(n, result, woff = math.inf, loff = -math.inf)
                    n = t.root
                    continue

            sims = n.sims
            if sims == 0:
                if n is not t.root:
                    simResult = Tree.simulate(g,n)
                    Tree.backprop(n, simResult)
                    n = t.root
                    continue

            num = n.amount()
            if num == 0:
                Tree.expand(g,n)
                first = n.children[0]
                n = first
                continue
            else:
                n = t.bestChildren(n,c,dtl)
                continue
    
    @staticmethod
    def expand(game, leaf):
        curBoard = leaf.state
        moves = curBoard.available()
        for move in moves:
            leaf.make(move)
    
    @staticmethod
    def simulate(game, leaf):
        original = leaf.player.opp
        simulateP = leaf.player.opp
        simulateB = deepcopy(leaf.state)

        while True:
            move = random.choice(simulateB.available())
            simulateB.update(move, simulateP.symbol)
            result = simulateB.winner()
            if result is not None:
                if result == original.symbol:
                    result = 1
                elif result == -1*original.symbol:
                    result = 2
                return result
            simulateP = simulateP.opp
    
    @staticmethod
    def backprop(leaf, result, woff=0, loff=0, doff=0):
        current = leaf
        original = leaf.player.opp.index

        while current is not None:
            current.sims += 1
            if result is OUTCOME_WIN:
                if current.player.index is original:
                    current.wins += win + woff
                else:
                    current.wins += lose + loff
            if result is OUTCOME_LOSE:
                if current.player.index is not original:
                    current.wins += win + woff
                else:
                    current.wins += lose + loff
            if result is OUTCOME_DRAW:
                current.wins += draw + doff

            woff = loff = doff = 0
            current = current.parent
        
    @staticmethod
    def UCT(node, exp):
        w = node.wins
        s = node.sims
        n = node.parent.sims
        c = exp

        if s == 0:
            return math.inf
        else:
            return (w/s) + (c*math.sqrt(math.log(n)/s))

class Game:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.current = p1
        self.other = p2
        p1.opp = p2
        p2.opp = p1
        p1.index = 1
        p2.index = -1
        self.board = Board2()

    def reset(self):
        self.board = Board2(board=np.zeros((4,4,4)))

    def switch(self):
        if self.current == self.p1:
            self.current = self.p2
            self.other = self.p1
        else:
            self.current = self.p1
            self.other = self.p2

    #computer
    def play(self, dbg = False):
        while not self.board.over:
            #p1
            p1Move = self.p1.action(self, dbg)
            if p1Move is None:
                print('mcts run out of time p1')
            else:
                self.board.update(p1Move, self.p1.symbol)
                self.show()

            if self.board.winner() is None:
                self.switch()
                p2Move = self.p2.action(self, dbg)
                if p2Move is None:
                    print('mcts run out of time p2')
                else:
                    self.board.update(p2Move, self.p2.symbol)
                    self.show()
                    if self.board.winner() is None:
                        self.switch()
                    else:
                        print('p2 wins')
            else:
                print('p1 wins')

    def playHuman(self, dbg = False):
        while not self.board.over:
            #p1
            p1Move = self.p1.action(self, dbg)
            if p1Move is None:
                print('mcts run out of time p1')
            else:
                self.board.update(p1Move, self.p1.symbol)
                self.show()
            if self.board.winner() is not None:
                if self.board.winner() == 1:
                    print("p1 Win")
                else:
                    print("Draw")
            else:
                self.switch()
                #change player
                p2Move = self.p2.action(self, dbg)
                if p2Move is None:
                    print('mcts run out of time p2')
                else:
                    self.board.update(p2Move, self.p2.symbol)
                    self.show()

                if self.board.winner() is not None:
                    if self.board.winner() == -1:
                        print("p2 Win")
                    else:
                        print("Draw")
                else:
                    self.switch()


    def show(self):
        k = 0
        self.grid_size = 4
        grid_range = range(self.grid_size)
        grid_output = []

        self.grid_data = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    self.grid_data.append(self.board.board[j,k,i])
        print(len(self.grid_data))
        j = 0
        i = 0
        k = 0
        for j in range(4):
            row_top = ' '*(self.grid_size*2+1) + '_'*(self.grid_size*4)
            if j:
                row_top = '|' + row_top[:self.grid_size*2-1] + '|' + '_'*(self.grid_size*2) + '|' + '_'*(self.grid_size*2-1) + '|'
            grid_output.append(row_top)

            for i in range(4):
                row_display = ' '*(self.grid_size*2-i*2) + '/' + ''.join((' ' + str(self.grid_data[k+x]).ljust(1) + ' /') for x in grid_range)
                k += self.grid_size
                row_bottom = ' '*(self.grid_size*2-i*2-1) + '/' + '___/'*self.grid_size

                if j != grid_range[-1]:
                    row_display += ' '*(i*2) + '|'
                    row_bottom += ' '*(i*2+1) + '|'
                if j:
                    row_display = row_display[:self.grid_size*4+1] + '|' + row_display[self.grid_size*4+2:]
                    row_bottom = row_bottom[:self.grid_size*4+1] + '|' + row_bottom[self.grid_size*4+2:]

                    row_display = '|' + row_display[1:]
                    row_bottom = '|' + row_bottom[1:]

                grid_output += [row_display, row_bottom]
        text = '\n'.join(grid_output)
        print(text)

import numpy as np
class Board2:
    def __init__(self, board=np.zeros((4,4,4))):
        self.board = board
        self.over = False

    def winner(self):
        #row 16
        #column 16 y z 16
        for i in range(4):
            for j in range(4):
                if self.board[i,j,0] + self.board[i,j,1] + self.board[i,j,2] + self.board[i,j,3] == 4:
                    self.over = True
                    return 1
                if self.board[i,j,0] + self.board[i,j,1] + self.board[i,j,2] + self.board[i,j,3] == -4:
                    self.over = True
                    return -1
                if self.board[i,0,j] + self.board[i,1,j] + self.board[i,2,j] + self.board[i,3,j] == 4:
                    self.over = True
                    return 1
                if self.board[i,0,j] + self.board[i,1,j] + self.board[i,2,j] + self.board[i,3,j] == -4:
                    self.over = True
                    return -1
                if self.board[0,i,j] + self.board[1,i,j] + self.board[2,i,j] + self.board[3,i,j] == 4:
                    self.over = True
                    return 1
                if self.board[0,i,j] + self.board[1,i,j] + self.board[2,i,j] + self.board[3,i,j] == -4:
                    self.over = True
                    return -1

        #diagonal xy 4 xz 4 yz 4
        #anti diagonal 12
        for i in range(4):
            if self.board[0,i,0] + self.board[1,i,1] + self.board[2,i,2] + self.board[3,i,3] == 4:
                self.over = True
                return 1
            if self.board[0,i,0] + self.board[1,i,1] + self.board[2,i,2] + self.board[3,i,3] == -4:
                self.over = True
                return -1
            if self.board[i,0,0] + self.board[i,1,1] + self.board[i,2,2] + self.board[i,3,3] == 4:
                self.over = True
                return 1
            if self.board[i,0,0] + self.board[i,1,1] + self.board[i,2,2] + self.board[i,3,3] == -4:
                self.over = True
                return -1
            if self.board[0,0,i] + self.board[1,1,i] + self.board[2,2,i] + self.board[3,3,i] == 4:
                self.over = True
                return 1
            if self.board[0,0,i] + self.board[1,1,i] + self.board[2,2,i] + self.board[3,3,i] == -4:
                self.over = True
                return -1

            if self.board[0,i,3] + self.board[1,i,2] + self.board[2,i,1] + self.board[3,i,0] == 4:
                self.over = True
                return 1
            if self.board[0,i,3] + self.board[1,i,2] + self.board[2,i,1] + self.board[3,i,0] == -4:
                self.over = True
                return -1
            if self.board[i,0,3] + self.board[i,1,2] + self.board[i,2,1] + self.board[i,3,0] == 4:
                self.over = True
                return 1
            if self.board[i,0,3] + self.board[i,1,2] + self.board[i,2,1] + self.board[i,3,0] == -4:
                self.over = True
                return -1
            if self.board[0,3,i] + self.board[1,2,i] + self.board[2,1,i] + self.board[3,0,i] == 4:
                self.over = True
                return 1
            if self.board[0,3,i] + self.board[1,2,i] + self.board[2,1,i] + self.board[3,0,i] == -4:
                self.over = True
                return -1
        #diagonal 3d 4
        if self.board[0,0,0] + self.board[1,1,1] + self.board[2,2,2] + self.board[3,3,3] == 4:
            self.over = True
            return 1
        if self.board[0,0,0] + self.board[1,1,1] + self.board[2,2,2] + self.board[3,3,3] == -4:
            self.over = True
            return -1
        if self.board[0,0,3] + self.board[1,1,2] + self.board[2,2,1] + self.board[3,3,0] == 4:
            self.over = True
            return 1
        if self.board[0,0,3] + self.board[1,1,2] + self.board[2,2,1] + self.board[3,3,0] == -4:
            self.over = True
            return -1
        if self.board[3,0,0] + self.board[2,1,1] + self.board[1,2,2] + self.board[0,3,3] == 4:
            self.over = True
            return 1
        if self.board[3,0,0] + self.board[2,1,1] + self.board[1,2,2] + self.board[0,3,3] == -4:
            self.over = True
            return -1
        if self.board[0,3,0] + self.board[1,2,1] + self.board[2,1,2] + self.board[3,0,3] == 4:
            self.over = True
            return 1
        if self.board[0,3,0] + self.board[1,2,1] + self.board[2,1,2] + self.board[3,0,3] == -4:
            self.over = True
            return -1

        if len(self.available()) == 0:
            self.over = True
            return 0 #draw
        
        self.over = False
        return None

        
    def available(self):
        return [(i,j,k) for i in range(4) for j in range(4) for k in range(4) if self.board[i,j,k] == 0]

    def update(self, move, player):
        self.board[move] = player
        return self

class HumanPlayer:
    def __init__(self, symbol):
        self.index = 0
        self.symbol = symbol
    def action(self, game, dbg = False):
        while True:
            string = []
            x = input()
            for y in x:
                if y.isdigit():
                    string.append(y)
            action = tuple(map(int, string))
            if action in game.board.available():
                return action

from datetime import datetime
class MCTSPlayer:
    def __init__(self, symbol, limit = 5000, exp = math.sqrt(2)):
        self.index = 0
        self.symbol = symbol
        self.limit = limit
        self.uct = exp
        self.opp = None

    def action(self, game, dbg = False, dtl = False):
        before = time.time()
        board = game.board.board
        match = []
        #try
        for i in range(4):
            for j in range(4):
                x = [board[i,j,0], board[i,j,1], board[i,j,2], board[i,j,3]]
                if (sum(x) == 3 or sum(x) == -3) and 0 in x:
                    for k in range(4):
                        if x[k] == 0:
                            return (i,j,k)
                if (sum(x) == 2 or sum(x) == -2 )and 0 in x:
                    mv = np.random.choice([k for k in range(4) if x[k] == 0])
                    match.append((i,j,mv))
                y = [board[i,0,j], board[i,1,j], board[i,2,j], board[i,3,j]]
                if (sum(y) == 3 or sum(y) == -3) and 0 in y:
                    for k in range(4):
                        if y[k] == 0:
                            return (i,k,j)
                if (sum(y) == 2 or sum(y) == -2) and 0 in y:
                    mv = np.random.choice([k for k in range(4) if y[k] == 0])
                    match.append((i,mv,j))
                z = [board[0,i,j], board[1,i,j], board[2,i,j], board[3,i,j]]
                if (sum(z) == 3 or sum(z) == -3) and 0 in z:
                    for k in range(4):
                        if z[k] == 0:
                            return (k,i,j)
                if (sum(z) == 2 or sum(z) == -2) and 0 in z:
                    mv = np.random.choice([k for k in range(4) if z[k] == 0])
                    match.append((mv,i,j))

        #diagonal xy 4 xz 4 yz 4
        #anti diagonal 12
        for i in range(4):
            xz = [board[0,i,0], board[1,i,1], board[2,i,2], board[3,i,3]]
            if (sum(xz) == 3 or sum(xz) == -3) and 0 in xz:
                for k in range(4):
                    if xz[k] == 0:
                        return (k,i,k)
            if (sum(xz) == 2 or sum(xz) == -2) and 0 in xz:
                mv = np.random.choice([k for k in range(4) if xz[k] == 0])
                match.append((mv,i,mv))
            yz = [board[i,0,0], board[i,1,1], board[i,2,2], board[i,3,3]]
            if (sum(yz) == 3 or sum(yz) == -3) and 0 in yz:
                    for k in range(4):
                        if yz[k] == 0:
                            return (i,k,k)
            if (sum(yz) == 2 or sum(yz) == -2) and 0 in yz:
                mv = np.random.choice([k for k in range(4) if yz[k] == 0])
                match.append((i,mv,mv))
            xy = [board[0,0,i], board[1,1,i], board[2,2,i], board[3,3,i]]
            if (sum(xy) == 3 or sum(xy) == -3) and 0 in xy:
                    for k in range(4):
                        if xy[k] == 0:
                            return (k,k,i)
            if (sum(xy) == 2 or sum(xy) == -2) and 0 in xy:
                mv = np.random.choice([k for k in range(4) if xy[k] == 0])
                match.append((mv,mv, i))

            xz = [board[0,i,3], board[1,i,2], board[2,i,1], board[3,i,0]]
            if (sum(xz) == 3 or sum(xz) == -3) and 0 in xz:
                    for k in range(4):
                        if xz[k] == 0:
                            return (k,i,3-k)
            if (sum(xz) == 2 or sum(xz) == -2) and 0 in xz:
                mv = np.random.choice([k for k in range(4) if xz[k] == 0])
                match.append((mv,i,3-mv))
            yz = [board[i,0,3], board[i,1,2], board[i,2,1], board[i,3,0]]
            if (sum(yz) == 3 or sum(yz) == -3) and 0 in yz:
                    for k in range(4):
                        if yz[k] == 0:
                            return (i,k,3-k)
            if (sum(yz) == 2 or sum(yz) == -2) and 0 in yz:
                mv = np.random.choice([k for k in range(4) if yz[k] == 0])
                match.append((i,mv,3-mv))
            xy = [board[0,3,i], board[1,2,i], board[2,1,i], board[3,0,i]]
            if (sum(xy) == 3 or sum(xy) == -3) and 0 in xy:
                    for k in range(4):
                        if xy[k] == 0:
                            return (k,3-k,i)
            if (sum(xy) == 2 or sum(xy) == -2) and 0 in xy:
                mv = np.random.choice([k for k in range(4) if xy[k] == 0])
                match.append((mv,3-mv, i))

        #diagonal 3d 4
        xyz = [board[0,0,0], board[1,1,1], board[2,2,2], board[3,3,3]]
        if (sum(xyz) == 3 or sum(xyz) == -3) and 0 in xyz:
                    for k in range(4):
                        if xyz[k] == 0:
                            return (k,k,k)
        if (sum(xyz) == 2 or sum(xyz) == -2) and 0 in xyz:
            mv = np.random.choice([k for k in range(4) if xyz[k] == 0])
            match.append((mv,mv,mv))
        xyz = [board[0,0,3], board[1,1,2], board[2,2,1], board[3,3,0]]
        if (sum(xyz) == 3 or sum(xyz) == -3) and 0 in xyz:
                    for k in range(4):
                        if xyz[k] == 0:
                            return (k,k,3-k)
        if (sum(xyz) == 2 or sum(xyz) == -2) and 0 in xyz:
            mv = np.random.choice([k for k in range(4) if xyz[k] == 0])
            match.append((mv,mv,3-mv))
        xyz = [board[3,0,0], board[2,1,1], board[1,2,2], board[0,3,3]]
        if (sum(xyz) == 3 or sum(xyz) == -3) and 0 in xyz:
                    for k in range(4):
                        if xyz[k] == 0:
                            return (3-k,k,k)
        if (sum(xyz) == 2 or sum(xyz) == -2) and 0 in xyz:
            mv = np.random.choice([k for k in range(4) if xyz[k] == 0])
            match.append((3-mv,mv,mv))
        xyz = [board[0,3,0], board[1,2,1], board[2,1,2], board[3,0,3]]
        if (sum(xyz) == 3 or sum(xyz) == -3) and 0 in xyz:
                    for k in range(4):
                        if xyz[k] == 0:
                            return (k,3-k,k)
        if (sum(xyz) == 2 or sum(xyz) == -2) and 0 in xyz:
            mv = np.random.choice([k for k in range(4) if xyz[k] == 0])
            match.append((mv,3-mv,mv))
        
        if len(match) > 0:
            return match[np.random.choice(len(match))]

        ms = time.time() - before
        self.limit = 4980 - ms*1000.
        tree = Tree.search(
            game = game,
            limit = self.limit,
            constant = self.uct,
            dbg = dbg,
            dtl = dtl
        )
        return tree.bestMove(dbg)

class RandomPlayer():
    def __init__(self, symbol):
        self.index = 0
        self.symbol = symbol
        self.opp = None
    def action(self, game, dbg):
        move = game.board.available()
        print(move)
        return move[np.random.choice(len(move))]
class Me():
    def __init__(self,symbol):
        self.index = 0
        self.symbol = symbol
        self.opp = None

# p1 = MCTSPlayer(1)
# p2 = MCTSPlayer(-1)
# game = Game(p1,p2)
# print(p1.opp.index)
# print(p2.opp.index)
# game.reset()
# game.play(dbg = False)
# p1 = MCTSPlayer(1)
# p2 = RandomPlayer(-1)
# game = Game(p1,p2)
# game.reset()
# game.play(dbg = False)

