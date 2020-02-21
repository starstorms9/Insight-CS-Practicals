'''
Simple AI agent working
Allows arbitrary NxN boards sizes
'''
import numpy as np

class tictactoe() :
    def __init__(self, players, boarddim) :
        self.boarddims = (boarddim, boarddim)
        self.turn = 1
        self.board = np.zeros((self.boarddims))
        self.players = players

    def play(self) :
        self.turn = 1
        winner = 0
        while winner == 0 :
            self.showBoard(self.board)
            
            if self.players == 2 :
                spot = self.getInput(self.turn)
            elif self.players == 1 :
                if self.turn == 1 :
                    spot = self.getInput(self.turn)
                else :
                    spot = self.getAISpot()
                    spotslose = self.playerLoseSpots(2, self.board)
                    spotswin = self.playerWinSpots(2, self.board)
                    print('\nSpots AI would lose: {}\nSpots AI would win:  {}'.format(spotslose, spotswin))
                    print('AI player has chosen spot {}'.format(spot))
                
            self.board[spot] = self.turn
            winner = self.checkWinner(self.board)
            self.turn = 1 if self.turn == 2 else 2
            
        if winner == -1 :
            print('Nobody won, but I hope you all still had fun')
        else :
            print('Winner is {}!'.format('X' if winner == 1 else 'Y'))

    def getAISpot(self) :        
        spotswin = self.playerWinSpots(2, self.board)
        if len(spotswin) > 0 :
            return spotswin[0]     
        
        spotslose = self.playerLoseSpots(2, self.board)
        if len(spotslose) > 0 :
            return spotslose[0]  
        
        return self.getRandSpot()
    
    def mockPlayInSpot(self, board, player, spot) :
        board = board.copy()
        board[spot] = player
        return board
    
    def playerWinSpots(self, player, board) :
        winspots = []
        board = board.copy()
        
        for i in range(self.boarddims[0]) :
            for j in range(self.boarddims[0]) :
                spot = (i,j)
                if board[spot] > 0 :
                    continue
                if player == self.checkWinner( self.mockPlayInSpot(board, player, spot )) :
                    winspots.append(spot)
        return winspots
    
    def playerLoseSpots(self, player, board) :
        return self.playerWinSpots(1 if player == 2 else 2, board)
        
    def getRandSpot(self) :
        spot = (0,0)
        valid = False
        while valid == False :
            spot = (np.random.randint(0, self.boarddims[0]), np.random.randint(0, self.boarddims[1]) )
            if (self.board[spot[0], spot[1]] > 0):
                continue
            return spot 

    def spotToXY(self, spot) :
        spoty = spot % self.boarddims[0] 
        spotx = int(spot / self.boarddims[1])
        return (spotx, spoty)

    def getInput(self, player) :
        valid = False
        spot = -1
        player = 'X' if player == 1 else 'Y'
        while valid == False :
            spot = input("Player {} input play position: ".format(player))
            try : spot = int(spot)
            except :
                print('Not a valid entry, try again')
                continue
                
            if (spot < 0 or spot >= (self.boarddims[0] * self.boarddims[1])) :
                print('Invalid spot, chose another')
                continue
            spot = self.spotToXY(spot)
            if (self.board[spot[0], spot[1]] > 0):
                print('Spot taken, chose another')
                continue
            valid = True
        return spot
    
    def showBoard(self, board) :
        index = 0
        for row in range(self.boarddims[0]) :
            for col in range(self.boarddims[1]) :
                position_entry = index if board[row,col] == 0 else ('X' if board[row,col] == 1 else 'Y') 
                print(' {} '.format(position_entry), end='')
                index = index + 1
            print('\n')
        
    def checkWinner(self, board) :
        winner = 0
        board = board.copy()
                
        # check cols
        for i in range(self.boarddims[1]) :
            if np.all(board[:,i] == 1) :
                return 1
            if np.all(board[:,i] == 2) :
                return 2
            
        # check rows
        for i in range(self.boarddims[0]) :
            if np.all(board[i,:] == 1) :
                return 1
            if np.all(board[i,:] == 2) :
                return 2
            
        # check diags
        if np.all(board.diagonal() == 1) :
            return 1
        if np.all(board.diagonal() == 2) :
            return 2
        
        reverse_diag = np.array([board[i,self.boarddims[1]-1-i] for i in range(self.boarddims[0])])        
        if np.all(reverse_diag == 1) :
            return 1
        if np.all(reverse_diag == 2) :
            return 2
        
        maxed_out = not (board==0).any()
        if (maxed_out and winner == 0) :
            winner = -1
        
        return winner

'''
 0  1  2 

 3  4  5 

 6  7  8 
'''
    
#%% Test game
playing = True
while playing == True :
    game = tictactoe(1, 3)
    game.play()
    
    valid = False
    while valid == False :
        try : 
            playing = bool(int(input('Play again? (0 for no, anything other # for yes) ')))
            if playing == False :
                print('Goodbye!')
                break
        except : 
            valid = False
        valid = True