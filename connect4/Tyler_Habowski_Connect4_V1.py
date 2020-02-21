'''
Interrupted time due to talking to Ed from Machina Labs / Matt
Finished in ~100 minutes with the okay AI agent
'''
import numpy as np
import random as rn

class C4():    
    def __init__(self, rows, cols, players) :
        self.player2name = {0:'_', 1:'X', 2:'O'}
        self.board = np.zeros((rows, cols))
        self.turn = 1
        self.players = players
    
    def play(self) :
        winner = 0
        while winner == 0 :
            self.showBoard(self.board)
            
            if (self.players == 2 or (self.players == 1 and self.turn==1)) :
                sel_col = self.getInput(self.turn)
            else :
                sel_col = self.getAISpot(self.board)
                print('AI win spots ', self.getWinSpots(self.board, 2))
                print('AI lose spots ', self.getLoseSpots(self.board, 2))
                print('AI has chosen col: ' + str(sel_col))
            
            self.board = self.playCol(self.board, sel_col, self.turn)
            winner = self.checkWinner(self.board)
            
            if winner > 0 :
                print('Congrats! Player {} won.'.format(self.player2name[winner]))
                break
            if sum(self.checkValidPlays(self.board)) == 0 :
                print('Nobody won, but I hope you all had fun.')
                break            
            
            self.turn = 1 if self.turn == 2 else 2
    
    def getInput(self, player) :
        while True :
            sel_col = input('Player {} Select column: '.format(self.player2name[player]))            
            try : sel_col = int(sel_col)
            except :
                print('Invalid input, try again')
                continue            
            if (sel_col < 0 or sel_col >= self.board.shape[1]) :
                print('Column out of range, try again')
                continue 
            if ( np.count_nonzero(self.board[:,sel_col]) >= self.board.shape[1] ) :
                print('Column full, try again')
                continue             
            return sel_col
    
    def getAISpot(self, board) :
        losespots = self.getLoseSpots(board, 2)
        winspots = self.getWinSpots(board, 2)
        if len(winspots) > 0 : 
            return winspots[0]
        elif len(losespots) > 0 :
            return losespots[0]
        else:
            return self.getRandValidPlay(board)
    
    def getWinSpots(self, board, player) :
        winspots = []
        for i in range(board.shape[1]):
            if (self.checkWinner(self.playCol(board, i, player)) == player) :
                winspots.append(i)
        return winspots
                
    def getLoseSpots(self, board, player) :
        return self.getWinSpots(board, 1 if player == 2 else 2)
    
    def getRandValidPlay(self, board) :
        valid_cols = self.checkValidPlays(board)
        return rn.choice(np.argwhere(valid_cols==True))[0]
    
    def getNextRow(self, board, col):
        return board.shape[0] - np.count_nonzero(board[:,col]) - 1
    
    def playCol(self, board, col, player) :
        test_board = board.copy()
        next_row = self.getNextRow(test_board, col)
        test_board[next_row,col] = player
        return test_board
    
    def checkValidPlays(self, board) :
        valid_cols = [self.checkValidPlay(board, i) for i in range(self.board.shape[1])]
        return np.array(valid_cols)
    
    def checkValidPlay(self, board, sel_col) :
        if (sel_col < 0 or sel_col >= self.board.shape[1]) :
            return False
        if ( np.count_nonzero(self.board[:,sel_col]) >= self.board.shape[1] ) :
            return False
        return True
    
    def showBoard(self, board='') :
        if board == '' : board = self.board
        col_labels = '\n '
        for i in range(board.shape[1]) : col_labels += '{}  '.format(i)
        print(col_labels)
        for r in range(board.shape[0]) :
            for c in range(board.shape[1]) :
                print(' {} '.format( self.player2name[board[r,c]] ), end='')
            print('\n')
    
    def checkWinner(self, board) :
        if self.checkColWinner(board, 1) or self.checkRowWinner(board, 1) or self.checkDiagWinner(board, 1) :
            return 1
        if self.checkColWinner(board, 2) or self.checkRowWinner(board, 2) or self.checkDiagWinner(board, 2) :
            return 2
        return 0
    
    def checkDiagWinner(self, board, player) :
        diagboards = self.get44Boards(board)
        for b44 in diagboards :
            if self.checkDiag(b44, player) :
                return True
        return False
        
    def checkDiag(self, board44, player) :
        diag = board44.diagonal()
        rdiag = board44[::-1,:].diagonal()
        return np.all(diag == player) or np.all(rdiag == player)
    
    def get44Boards(self, board) :
        boards = []
        for i in range(board.shape[0]-4+1) :
            for j in range(board.shape[1]-4+1) :
                boards.append(board[i:i+4:1,j:j+4:1])
        return boards
    
    def checkColWinner(self, board, player) :
        for c in range(board.shape[1]) :
            sequential = 0
            for r in range(board.shape[0]) :
                if board[r,c] == player :
                    sequential += 1
                else:
                    sequential = 0
            if sequential >= 4:
                return True
        return False
                
    def checkRowWinner(self, board, player) :
        for r in range(board.shape[0]) :
            sequential = 0
            for c in range(board.shape[1]) :
                if board[r,c] == player :
                    sequential += 1
                else:
                    sequential = 0
            if sequential >= 4:
                return True
        return False

#%% Test Game
playing = True
while playing :
    num_players = input('Number of players: ')
    game_rows = input('Number of rows: ')
    game_cols = input('Number of columns: ')
    if not (num_players.isdigit() and game_cols.isdigit() and game_rows.isdigit()):
        print('Invalid input params, try again.')
        continue
    game = C4(int(game_rows),int(game_cols),int(num_players))
    game.play()
    done = True
    if not (input('Done playing, play again? (y for yes, anything else for no)') == 'y') :
        break

