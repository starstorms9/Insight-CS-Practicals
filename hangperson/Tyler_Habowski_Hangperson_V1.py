'''
Created on Thu Feb 13 10:40:09 2020

@author: starstorms
'''
import urllib.request as request
import random as rn

class hangPerson() :
    def __init__(self, max_attempts) :
        self.words = self.Word()
        self.max_attempts = max_attempts
        self.guesses = []
     
    def reset(self) :
        self.guesses = []
        self.words.chooseNewSecret()
    
    def playMutli(self) :
        while True :
            self.playGame()
            playAgain = input('Want to play again? [y/n]')
            if (playAgain.lower() == 'y') :
                self.reset()
                continue
            else:
                print('Goodbye!')
                break
    
    def playGame(self) :
        while True :
            self.showGame()
            guess = self.getInput()
            self.guesses.append(guess)
            won, game_over = self.checkWin()
            if (won) :
                print('You won!')
                return
            if (game_over) :
                print('You lost! Secret word was: {}'.format(self.words.secretWord))
                return
            
    def getInput(self) :
        while True:
            guess = input('What is your guess? ')
            if (guess.isalpha() and len(guess)==1) :
                if (guess in self.guesses) :
                    print('Already guessed {}. Try again'.format(guess))
                    print('So far you have guessed: {}'.format(sorted(self.guesses)))
                    continue
                
                if (guess in self.words.secretWord) :
                    print('\nLucky guess, {} is in the secret word'.format(guess))
                else :
                    print('\nBad guess, {} is not in the secret word'.format(guess))
                
                return guess.lower()
            else:
                print('Invalid input, try again.')
    
    def checkWin(self) :
        won = True
        for sc in self.words.secretWord :
            if (not sc in self.guesses) :
                won = False
                break
        
        return won, (self.gotWrong() >= self.max_attempts)
    
    def gotWrong(self) :
        wrong = 0
        for g in self.guesses :
            if (not g in self.words.secretWord) :
                wrong += 1
        return wrong
    
    def showGame(self) :
        print('\n')
        for c in self.words.secretWord :
            print(' ' + c if c in self.guesses else ' _', end='')
        print('\nYou have {} guesses remaining.'.format(self.max_attempts - self.gotWrong()))
        
    class Word() :
        def __init__(self) :
            self.all_words = ['hello', 'break', 'noppe'] # self.getWords()
            self.chooseNewSecret()
        
        def getWords(self) :
            self.target_url = "https://www.norvig.com/ngrams/sowpods.txt"
            data = request.urlopen(self.target_url)
            words = list()
            for word in data:
                words.append(word.decode("utf-8").strip())
                
        def chooseNewSecret(self) :
            self.secretWord = rn.choice(self.all_words)
    
#%%
hp = hangPerson(4)
hp.playGame()