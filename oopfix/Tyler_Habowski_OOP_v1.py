'''

Interrupted by Noble.ai demo
In this OOP exercise, everyone must add at least three classes

'''
#%% Imports
import random
import re
import time
from string import ascii_lowercase

#%% Given code
def setupgrid(gridsize, start, numberofmines):
    emptygrid = [['0' for i in range(gridsize)] for i in range(gridsize)]

    mines = getmines(emptygrid, start, numberofmines)

    for i, j in mines:
        emptygrid[i][j] = 'X'

    grid = getnumbers(emptygrid)

    return (grid, mines)

def showgrid(grid):
    gridsize = len(grid)

    horizontal = '   ' + (4 * gridsize * '-') + '-'

    # Print top column letters
    toplabel = '     '

    for i in ascii_lowercase[:gridsize]:
        toplabel = toplabel + i + '   '

    print(toplabel + '\n' + horizontal)

    for idx, i in enumerate(grid):
        row = '{0:2} |'.format(idx + 1)

        for j in i:
            row = row + ' ' + j + ' |'

        print(row + '\n' + horizontal)

    print('')

def getrandomcell(grid):
    gridsize = len(grid)
    a = random.randint(0, gridsize - 1)
    b = random.randint(0, gridsize - 1)
    return (a, b)

def getneighbors(grid, rowno, colno):
    gridsize = len(grid)
    neighbors = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            elif -1 < (rowno + i) < gridsize and -1 < (colno + j) < gridsize:
                neighbors.append((rowno + i, colno + j))
    return neighbors

def getmines(grid, start, numberofmines):
    mines = []
    neighbors = getneighbors(grid, *start)

    for i in range(numberofmines):
        cell = getrandomcell(grid)
        while cell == start or cell in mines or cell in neighbors:
            cell = getrandomcell(grid)
        mines.append(cell)
    return mines

def getnumbers(grid):
    for rowno, row in enumerate(grid):
        for colno, cell in enumerate(row):
            if cell != 'X':
                # Gets the values of the neighbors
                values = [grid[r][c] for r, c in getneighbors(grid,
                                                              rowno, colno)]
                # Counts how many are mines
                grid[rowno][colno] = str(values.count('X'))
    return grid

def showcells(grid, currgrid, rowno, colno):
    # Exit function if the cell was already shown
    if currgrid[rowno][colno] != ' ':
        return

    # Show current cell
    currgrid[rowno][colno] = grid[rowno][colno]

    # Get the neighbors if the cell is empty
    if grid[rowno][colno] == '0':
        for r, c in getneighbors(grid, rowno, colno):
            # Repeat function for each neighbor that doesn't have a flag
            if currgrid[r][c] != 'F':
                showcells(grid, currgrid, r, c)

def playagain():
    choice = input('Play again? (y/n): ')

    return choice.lower() == 'y'

def playgame():
    gridsize = 9
    numberofmines = 10

    currgrid = [[' ' for i in range(gridsize)] for i in range(gridsize)]

    grid = []
    flags = []
    starttime = 0

    helpmessage = ("Type the column followed by the row (eg. a5). "
                   "To put or remove a flag, add 'f' to the cell (eg. a5f).")

    showgrid(currgrid)
    print(helpmessage + " Type 'help' to show this message again.\n")

    while True:
        minesleft = numberofmines - len(flags)
        prompt = input('Enter the cell ({} mines left): '.format(minesleft))
        result = parseinput(prompt, gridsize, helpmessage + '\n')

        message = result['message']
        cell = result['cell']

        if cell:
            print('\n\n')
            rowno, colno = cell
            currcell = currgrid[rowno][colno]
            flag = result['flag']

            if not grid:
                grid, mines = setupgrid(gridsize, cell, numberofmines)
            if not starttime:
                starttime = time.time()

            if flag:
                # Add a flag if the cell is empty
                if currcell == ' ':
                    currgrid[rowno][colno] = 'F'
                    flags.append(cell)
                # Remove the flag if there is one
                elif currcell == 'F':
                    currgrid[rowno][colno] = ' '
                    flags.remove(cell)
                else:
                    message = 'Cannot put a flag there'

            # If there is a flag there, show a message
            elif cell in flags:
                message = 'There is a flag there'

            elif grid[rowno][colno] == 'X':
                print('Game Over\n')
                showgrid(grid)
                if playagain():
                    playgame()
                return

            elif currcell == ' ':
                showcells(grid, currgrid, rowno, colno)

            else:
                message = "That cell is already shown"

            if set(flags) == set(mines):
                minutes, seconds = divmod(int(time.time() - starttime), 60)
                print(
                    'You Win. '
                    'It took you {} minutes and {} seconds.\n'.format(minutes,
                                                                      seconds))
                showgrid(grid)
                if playagain():
                    playgame()
                return

        showgrid(currgrid)
        print(message)

playgame()

#%% My code
'''
Class structure is:
    sweeper :
        game :
            board
            cell
            player
'''

class Sweeper() :
    def __init__(self) :
        self.cur_game = self.Game()
    
    def playagain(self):
        choice = input('Play again? (y/n): ')    
        return choice.lower() == 'y'
    
    def playGames(self) :
        self.cur_game.playgame()
        while True :
            if self.playagain() :
                self.cur_game = self.Game()
                self.cur_game.playgame()
            else :
                break
        print('Goodbye')

    class Game() :
        def __init__(self) :
            self.player = self.Player()
            self.board = self.Board()

        def playgame(self):
            starttime = 0
        
            helpmessage = ("Type the column followed by the row (eg. a5). "
                           "To put or remove a flag, add 'f' to the cell (eg. a5f).")
        
            self.board.showgrid()
            print(helpmessage + " Type 'help' to show this message again.\n")
        
            while True:
                minesleft = self.board.numberofmines - len(self.board.flags)
                prompt = input('Enter the cell ({} mines left): '.format(minesleft))
                result = self.player.parseinput(prompt, self.board.gridsize, helpmessage + '\n')
        
                message = result['message']
                cell = result['cell']
        
                if cell:
                    print('\n\n')
                    rowno, colno = cell
                    currcell = self.board.currgrid[rowno][colno]
                    flag = result['flag']
        
                    if not self.board.grid:
                        self.board.setupgrid(cell)
                    if not starttime:
                        starttime = time.time()
        
                    if flag:
                        # Add a flag if the cell is empty
                        if currcell == ' ':
                            self.board.currgrid[rowno][colno] = 'F'
                            self.board.flags.append(cell)
                        # Remove the flag if there is one
                        elif currcell == 'F':
                            self.board.currgrid[rowno][colno] = ' '
                            self.board.flags.remove(cell)
                        else:
                            message = 'Cannot put a flag there'
        
                    # If there is a flag there, show a message
                    elif cell in self.board.flags:
                        message = 'There is a flag there'
        
                    elif self.board.grid[rowno][colno] == 'X':
                        print('Game Over\n')
                        self.board.showgrid()
                        if playagain():
                            playgame()
                        return
        
                    elif currcell == ' ':
                        self.board.showcells(self.board.grid, self.board.currgrid, rowno, colno)
        
                    else:
                        message = "That cell is already shown"
        
                    if set(self.board.flags) == set(self.mines):
                        minutes, seconds = divmod(int(time.time() - starttime), 60)
                        print( 'You Win. ', 'It took you {} minutes and {} seconds.\n'.format(minutes, seconds))
                        self.board.showgrid()
                        if playagain():
                            playgame()
                        return
        
                showgrid(self.currgrid)
                print(message)

        class Player() :
            def __init(self) :
                pass
            
            def parseinput(self, inputstring, gridsize, helpmessage):
                cell = ()
                flag = False
                message = "Invalid cell. " + helpmessage
            
                pattern = r'([a-{}])([0-9]+)(f?)'.format(ascii_lowercase[gridsize - 1])
                validinput = re.match(pattern, inputstring)
            
                if inputstring == 'help':
                    message = helpmessage
            
                elif validinput:
                    rowno = int(validinput.group(2)) - 1
                    colno = ascii_lowercase.index(validinput.group(1))
                    flag = bool(validinput.group(3))
            
                    if -1 < rowno < gridsize:
                        cell = (rowno, colno)
                        message = ''
            
                return {'cell': cell, 'flag': flag, 'message': message}
    
        class Board() : 
            def __init__(self) :
                self.gridsize = 9
                self.numberofmines = 10            
                self.currgrid = [[' ' for i in range(self.gridsize)] for i in range(self.gridsize)]
                self.flags = []
                self.grid = []
             
            def getrandomcell(self):
                a = random.randint(0, self.gridsize - 1)
                b = random.randint(0, self.gridsize - 1)
                return (a, b)
            
            def setupgrid(self, start):
                emptygrid = [['0' for i in range(self.gridsize)] for i in range(self.gridsize)]        
                self.mines = self.getmines(start)
            
                for i, j in self.mines:
                    emptygrid[i][j] = 'X'
            
                self.grid = self.getnumbers()
                return self.grid
            
            def getmines(self, start):
                mines = []
                neighbors = self.getneighbors(*start)
            
                for i in range(self.numberofmines):
                    cell = self.getrandomcell()
                    while cell == start or cell in mines or cell in neighbors:
                        cell = self.getrandomcell()
                    mines.append(cell)
                return mines
            
            def showcells(self, grid, currgrid, rowno, colno):
                # Exit function if the cell was already shown
                if currgrid[rowno][colno] != ' ':
                    return
            
                # Show current cell
                currgrid[rowno][colno] = grid[rowno][colno]
            
                # Get the neighbors if the cell is empty
                if grid[rowno][colno] == '0':
                    for r, c in getneighbors(grid, rowno, colno):
                        # Repeat function for each neighbor that doesn't have a flag
                        if currgrid[r][c] != 'F':
                            showcells(grid, currgrid, r, c)
        
            def getnumbers(self):
                for rowno, row in enumerate(self.grid):
                    for colno, cell in enumerate(row):
                        if cell != 'X':
                            # Gets the values of the neighbors
                            values = [self.grid[r][c] for r, c in self.getneighbors(self.grid, rowno, colno)]
                            # Counts how many are mines
                            self.grid[rowno][colno] = str(values.count('X'))
                return self.grid
        
            def showgrid(self):            
                horizontal = '   ' + (4 * self.gridsize * '-') + '-'
            
                # Print top column letters
                toplabel = '     '
            
                for i in ascii_lowercase[:self.gridsize]:
                    toplabel = toplabel + i + '   '
            
                print(toplabel + '\n' + horizontal)
            
                for idx, i in enumerate(self.grid):
                    row = '{0:2} |'.format(idx + 1)
            
                    for j in i:
                        row = row + ' ' + j + ' |'
            
                    print(row + '\n' + horizontal)        
                print('')        
            
            def getneighbors(self, rowno, colno):
                neighbors = []
            
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        elif -1 < (rowno + i) < self.gridsize and -1 < (colno + j) < self.gridsize:
                            neighbors.append((rowno + i, colno + j))
                return neighbors

        
#%% Testing
game = Sweeper()
game.playGames()