import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import random

#    x = min + (max - min)*random.random()

population = random.randint(50)
#population = 10

original = np.arange(population) # random.rand
continuousScale = np.random.rand(population)

finalOpinions = [0]*population

opinions = []
finalOpinions55 = []

for i in range(population):
    x = round(continuousScale[i], 3)
    continuousScale[i] = x
    i += 1

def subject_and_neighbour(size, sample): # Defaunt

#    num = size
    global num
    global subject
    global leftNeighbour
    global rightNeighbour

    global Left       # having a left neighbour
    global Right      # having a right neighbour
    global case

    num = 0

    for i in range(size):

        if size > 2 and size < 100:

            if num > 0 and num+1 != size: # middle

                subject = sample[num]

                rightNeighbour = sample[num+1]

                leftNeighbour = sample[num-1]

                Left = True
                Right = True

            elif num == size: # maximum case

                subject = sample[num-1]

                leftNeighbour = sample[num - 2]
                rightNeighbour = 0

                Left = True
                Right = False

            elif num +1 == size:

                subject = sample[num]

                leftNeighbour = sample[num - 2]
                rightNeighbour = 0

                Left = True
                Right = False

            else: # 0th case

                subject = sample[num]

                leftNeighbour = 0
                rightNeighbour = sample[num+1]

                Left = False
                Right = True

        if size == 2:

            if num > 0:

                subject = sample[num]

                leftNeighbour = sample[num-1]
                rightNeighbour = 0

                Left = True
                Right = False

            else:

                subject = sample[num]

                leftNeighbour = 0
                rightNeighbour = sample[num-1]

                Left = False
                Right = True

        elif  size == 1:

            #case = 7

            subject = sample[num]
            print('person is', sample[num], '\n')
            print('There is only one person in the sample')

            Left = False
            Right = False

        elif size == 0 or size > 100:

            subject = 0
            print('corrupt smaple')

        num +=1 # next neighbour

        select_random_neighbour(subject)
        opinions.append(opinion)
        Threshold(opinion, 0.2, 0.2)

        #print(neighbourNum)

    print('\n final opinions :\n', finalOpinions)

def select_random_neighbour(subject):
    global opinion
    global neighbourNum
    global T
    global Beta
    global neighbour

    s = random.randint(2)
    neighbourNum = 1

    if population > 2:
        if Left == True and Right == True: #case 1

            if s == 0:  # left selected
                neighbourNum = num - 1
                neighbour = leftNeighbour
                opinion = subject - neighbour

            elif s == 1:  # right selected
                neighbourNum = num + 1
                neighbour = rightNeighbour
                opinion = subject - neighbour

        elif Left == True and Right == False: #case 2 & 3  # left selected
            neighbourNum = num - 1
            neighbour = leftNeighbour
            opinion = subject - neighbour

        elif Left == False and Right == True: #case 4  # right selected
            neighbourNum = num + 1
            neighbour = rightNeighbour
            opinion = subject - neighbour

    elif population == 2:

        if Left == True and Right == False: #case 5  left selected
            neighbourNum = num - 1
            neighbour = leftNeighbour
            opinion = subject - neighbour

        elif Left == False and Right == True: #case 6   right selected
            neighbourNum = num + 1
            neighbour = rightNeighbour
            opinion = subject - neighbour

    else:
        opinion = 0
        print('corrupt sample')

    opinion = round(opinion, 3)
def Threshold(value, T, beta):
    global subject2
    global neighbour2

    subject2 = 0
    neighbour2 = 0

    if abs(value) < T:
        subject2 = round(subject + beta*(neighbour - subject), 3)
        neighbour2 = round(subject + beta*(subject - neighbour), 3)

    else:
        subject2 = subject     # same as continuous sample
        neighbour2 = neighbour

    finalOpinions[num-1] = subject2
    finalOpinions[neighbourNum -1] = neighbour2

def test_defaunt(size, sample):
    print('population is', population, '\n')

    print('original sample: ')
    print(original, '\n')

    print('continuousScale sample ')
    print(continuousScale, '\n')

    subject_and_neighbour(size, sample)

    print('\n opinions :\n', opinions)
    print('\n final opinions :\n', finalOpinions)

subject_and_neighbour(population, continuousScale)





