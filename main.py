import numpy as np

from random import shuffle, choice, randint

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Game:

    def __init__(self, generation):
        self.generation = generation

        self.deck = []
        self.queue = []

        self.trumpCard = None

        self.players = None

        self.startingPlayer = 1

        for suit in range(4):
            for number in range(10):
                self.deck.append(Card(suit, number))
        shuffle(self.deck)

    def setPlayers(self, players):
        self.players = players

    def evaluateQueue(self):

        trumps = []
        priority = []

        for card in self.queue:
            if card.suit == self.trumpCard.suit:
                trumps.append(int(card))
            else:
                trumps.append(-1)

        if len(trumps) != 0:
            return self.queue[trumps.index(max(trumps))]

        else:
            for card in self.queue:

                if card.suit == self.queue[0].suit:
                    priority.append(int(card))

                else:
                    priority.append(-1)

            return self.queue[priority.index(max(priority))]

    def determineWinners(self):
        pointsList = []

        for player in self.players:
            pointsList.append(player.countPoints())

        #print(pointsList)

        if pointsList[0] + pointsList[2] > pointsList[1] + pointsList[3]:
            #print(self.players[0], self.players[2], sep='\n')
            return self.players[0], self.players[2]
        elif pointsList[1] + pointsList[3] > pointsList[0] + pointsList[2]:
            #print(self.players[1], self.players[3], sep='\n')
            return self.players[1], self.players[3]

    def play(self):
        self.trumpCard = self.deck.pop(0)
        self.deck.append(self.trumpCard)

        for card in self.deck:
            if card.suit == self.trumpCard.suit:
                card.trump = True

        print(
            '\n\n                   GAME '+
            str(self.generation)+
            '\n---------------------------------------------------\n\n'
        )

        print('Brisc suit: ' + str(self.trumpCard.suitStr()))


        while len(self.deck) != 0:

            for player in self.players[self.startingPlayer - 1 :] + \
                          self.players[: self.startingPlayer - 1]:


                player.updateInputs()
                player.normalizeInputs()

                player.ai.run()

                player.recordData()

                if self.generation >= 900:

                    input('\nenter to continue\n')

            for player in self.players:

                if self.evaluateQueue() == player.previousCard:
                    #player won queue!

                    player.wonQueues.append(self.queue)



                    if self.generation >= 900:

                        print(str(player) + 'won: ')
                        for card in self.queue:
                            print(card)

                        input('\nenter to continue\n')

                    #print('training... generation ' + str(self.generation))

                    #player.ai.train()

                    self.startingPlayer = self.players.index(player) + 1

                    break

            self.queue = []

            for player in self.players:
                player.drawCard()

        winners = self.determineWinners()

        if winners:

            winners[0].ai.train()
            winners[1].ai.train()

            if self.generation >= 900:

                print('Winners: ' + str(winners[0]), str(winners[1]), sep=', ', end='\n')

        for player in self.players:
            player.clearData()


class AI:

    def __init__(self, player, inputSize, randomGames):
        self.player = player
        self.inputSize = inputSize
        self.randomGames = randomGames

        self.playedCard = None

        self.model = Sequential([
            Dense(300, input_shape=(inputSize,), activation='relu'),
            Dense(500, activation='relu'),
            Dense(500, activation='relu'),
            Dense(500, activation='relu'),
            Dense(3, activation='softmax')
        ])

        self.model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def countPoints(self, queue):
        points = 0
        for card in queue:
            points += card.points
        return points

    def train(self):

        ins = np.asarray(self.player.gameInputs[:])
        outs = np.asarray(self.player.gameOutputs[:])

        self.model.fit(ins,
                  outs,
                  batch_size=1,
                  epochs=15,
                  shuffle=True,
                  verbose=0,
        )

    def run(self):

        ins = np.asarray([np.asarray(self.player.inputs[:])])

        if self.player.game.generation > self.randomGames:
            outputs = self.model.predict(ins, batch_size=1, verbose=0)[0]
            prediction = list(outputs).index(max(outputs))
        else:
            prediction = randint(0, 2)
            outputs = np.zeros(3)
            outputs[prediction] = 1

        self.player.lastOutput = outputs
        if self.player.game.generation >= 900:
            print(str(self.player) + ' plays the ' + str(self.player.hand[prediction]))
        self.player.playCard(self.player.hand[prediction])

class Player:

    def __init__(self, playerNumber):
        self.game = None
        self.playerNumber = playerNumber
        self.teammate = None
        self.opponents = []
        self.ai = None
        self.hand = []
        self.previousCard = None
        self.lastOutput = []
        self.gameInputs = []
        self.gameOutputs = []
        self.wonQueues = []

        self.inputs = []
        self.inputRangeMap = [
            #card 1 brisc
            (0, 1),
            #card 1 priority
            (0, 1),
            #card 1 number
            (0, 9),
            #card 2 brisc
            (0, 1),
            #card 2 priority
            (0, 1),
            #card 2 number
            (0, 9),
            #card 3 brisc
            (0, 1),
            #card 3 priority
            (0, 1),
            #card 3 number
            (0, 9),
            #queue size
            (0, 3),
            #queue points
            (0, 33),
            #high card brisc
            (0, 1),
            #high card number
            (0, 9)
        ]

    def __str__(self):
        cardStrings = []
        for card in self.hand:
            cardStrings.append(str(card))
        return 'Player ' + str(self.playerNumber) + ': ' + str(cardStrings)

    def newGame(self, game):
        self.game = game
        self.hand = []
        self.previousCard = None
        self.lastOutput = []
        self.wonQueues = []
        self.inputs = []

        for i in range(3):
            self.drawCard()

    def setTeammate(self, teammate):
        self.teammate = teammate

    def setOpponents(self, opponents):
        self.opponents = opponents

    def setGame(self, game):
        self.game = game

    def setAI(self, ai):
        self.ai = ai

    def playCard(self, card):
        self.previousCard = card
        self.game.queue.append(self.hand.pop(self.hand.index(card)))

    def drawCard(self):
        self.hand.append(self.game.deck.pop(0))

    def updateInputs(self):

        self.inputs = []

        for card in self.hand:

            if card.trump:
                self.inputs.append(1)
            else:
                self.inputs.append(0)

            if len(self.game.queue) == 0 or card.suit == self.game.queue[0].suit:
                self.inputs.append(1)
            else:
                self.inputs.append(0)

            self.inputs.append(card.number)

        if len(self.game.queue) == 0:
            for x in range(4):
                self.inputs.append(0)

        else:
            self.inputs.append(len(self.game.queue))

            self.inputs.append(self.queuePoints())

            highestCard = self.game.evaluateQueue()

            if highestCard.trump:
                self.inputs.append(1)
            else:
                self.inputs.append(0)

            self.inputs.append(highestCard.number)

    def normalizeInputs(self):

        for i in range(len(self.inputs)):
            self.inputs[i] = np.interp(self.inputs[i], self.inputRangeMap[i], (0, 1))

    def recordData(self):
        self.gameInputs.append(np.asarray(self.inputs))
        self.gameOutputs.append(np.asarray(self.lastOutput))

    def clearData(self):
        self.gameInputs = []
        self.gameOutputs = []

    def countPoints(self):
        points = 0

        for queue in self.wonQueues:
            for card in queue:
                points += card.points

        return points

    def queuePoints(self):
        points = 0
        for card in self.game.queue:
            points += card.points
        return points

class Card:

    def __init__(self, suit, number):
        self.suit = suit
        self.number = number
        self.trump = False
        self.points = 0

        self.suitNames = ['Coins', 'Swords', 'Cups', 'Clubs']
        self.numberNames = ['Two', 'Four', 'Five', 'Six', 'Seven',
                       'Jack', 'Knight', 'King', 'Three', 'Ace']

        if number == 5:
            self.points = 2
        elif number == 6:
            self.points = 3
        elif number == 7:
            self.points = 4
        elif number == 8:
            self.points = 10
        elif number == 9:
            self.points = 11

    def __eq__(self, other):
        return self.suit == other.suit and self.number == other.number

    def __str__(self):
        return self.numberNames[self.number] + ' of ' + self.suitNames[self.suit]

    def __int__(self):
        return self.number

    def suitStr(self):
        return self.suitNames[self.suit]

player1 = Player(1)
player2 = Player(2)
player3 = Player(3)
player4 = Player(4)

ai1 = AI(player1, 13, 50)
ai2 = AI(player2, 13, 50)
ai3 = AI(player3, 13, 50)
ai4 = AI(player4, 13, 50)

player1.setAI(ai1)
player2.setAI(ai2)
player3.setAI(ai3)
player4.setAI(ai4)

player1.setTeammate(player3)
player1.setOpponents((player2, player4))

player2.setTeammate(player4)
player2.setOpponents((player1, player3))

player3.setTeammate(player1)
player3.setOpponents((player2, player4))

player4.setTeammate(player2)
player4.setOpponents((player1, player3))


for gen in range(1000):
    game = Game(gen)

    player1.newGame(game)
    player2.newGame(game)
    player3.newGame(game)
    player4.newGame(game)

    game.setPlayers([player1, player2, player3, player4])

    game.play()
