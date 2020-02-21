'''
Started 13:48
'''

import numpy as np
import os
import random as rn
import time

class Sweeper() :
    def __init__(self, rows=5, cols=5) :
        self.board = self.Board(rows,cols)
        
    def playGame(self, ) :
        pass
    
    def playGames(self, ) :
        pass
    
    def getInput(self, ) :
        return 0
    
    def showGame(self) :
        self.board.show()
    
    def checkWin(self, ) :
        return 0
        
    class Board() :
        def __init__(self, rows, cols, num_bombs) :
            self.rows = rows
            self.cols = cols
            self.numbombs = num_bombs
            self.board = [ ([0] * cols) for row in range(rows) ]            
            for r in range(rows) :
                for c in range(cols) :
                    self.board[r][c] = self.Space(r, c)
            
            self.placeBombs(num_bombs)
            self.getAllNeigbors()
            self.updateBombDists()
            
        def placeBombs(self, num_bombs) :
            placed_bombs = 0
            while placed_bombs < num_bombs :
                spot = np.random.randint((self.rows, self.cols))
                test_space = self.getSpace(spot)
                
                if (not test_space.isbomb) :
                    test_space.isbomb = True   
                    test_space.bombsnearby = -1
                    placed_bombs += 1
        
        def getSpace(self, spot) :
            return self.board[spot[0]][spot[1]]
                    
        def show(self) :
            for r in range(self.rows) :
                for c in range(self.cols) :
                    print(self.board[r][c], end=' ')
                print('\n')
        
        def updateBombDists(self) :
            for r in range(self.rows) :
                for c in range(self.cols) :
                    self.getSpace((r,c)).nbombs = self.getSpace((r,c)).neighborBombs()
        
        '''
        Return a list with neighbor space links (null if doesn't exist):
        0  1  2
        3     4
        5  6  7
        '''
        def getAllNeigbors(self) :
            for r in range(self.rows) :
                for c in range(self.cols) :
                    neighbors = []                    
                    for relrow in [-1, 0, 1] :
                        for relcol in [-1, 0, 1] :
                            if (relrow==0 and relcol==0) :
                                continue
                            row = r + relrow                            
                            col = c + relcol
                            if (row > self.rows or row < 0) or (col > self.cols or col < 0):
                                neighbors.append(None)
                            else:
                                neighbors.append(self.getSpace((relrow,relcol)))
                    self.getSpace((r,c)).neighbors = neighbors
        
        def propogate(self, space) :
            pass
    
        class Space() :
            def __init__(self, row, col) :
                self.isbomb = False
                self.revealed = False            
                self.flagged = False
                self.bombsnearby = 0
                self.row = row
                self.col = col
                self.neighbors = []
            
            def __repr__(self) :
                return 'B' if self.isbomb else str(self.bombsnearby)
            
            def neighborBombs(self) :
                nbombs = 0
                for neighbor in self.neighbors :
                    if neighbor == None :
                        continue
                    if neighbor.isbomb :
                        nbombs += 1
                return nbombs
            
            def printNeighbor(self) :
                print('{} {} {}'.format(self.neighbors[0], self.neighbors[1], self.neighbors[2]))
                print('{} {} {}'.format(self.neighbors[3], ' ', self.neighbors[4]))
                print('{} {} {}'.format(self.neighbors[5], self.neighbors[6], self.neighbors[7]))
        
#%% Testing
board = Sweeper.Board(3,3,2)
board.show()

board.getSpace((1,1)).printNeighbor()
board.getSpace((0,1)).printNeighbor()
board.getSpace((1,0)).printNeighbor()
