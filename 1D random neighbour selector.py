import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import random

population = random.randint(50)
#population = 4

print('population is', population, '\n')

original = []


original = np.arange(population)
binary = np.arange(population)

print('original sample: ')
print(original, '\n')

array = []

for i in range(population):
#    print(sample[i])
    x = random.randint(2)
    binary[i] = x
    i += 1
print('binary sample ')
print(binary, '\n')

def subject_and_neighbour(size, sample):
    num = random.randint(size)
#    num = size
    global subject
    global leftNeighbour
    global rightNeighbour
    global Left
    global Right
    global case
    global opinion



    print('size =', size)
    print('num =', num)


    if size > 2 and size < 100:

        if num > 0 and num != size:
            print('greater than 0')
            print('subject number is', num+1)
            print('person', num+1, 'is', sample[num], '\n')

        elif num == size:
            print('greater than 0')
            print('subject number is', num )
            print('person', num , 'is', sample[num-1], '\n')

        else:
            print('less than 0')
            print('subject number is', num+1 )
            print('person', num+1, 'is', sample[num], '\n')


        if num > 0 and num+1 != size: # middle
            print('middle')
            case = 1

            subject = sample[num]
            print('person', num+1, 'is', sample[num])

            rightNeighbour = sample[num+1]
            print('The right neighbor of person', num+1, 'is', sample[num+1] )

            leftNeighbour = sample[num-1]
            print('The left neighbor of person', num+1, 'is', sample[num-1])

            Left = True
            Right = True

        elif num == size: # maximum case
            print('maximum')
            case = 2

            subject = sample[num-1]
            print('person', num, 'is', sample[num-1])

            leftNeighbour = sample[num - 2]
            righttNeighbour = 0
            print('The left neighbor of person', num, 'is', sample[num-2])

            Left = True
            Right = False

            print('There is no right neighbour')

        elif num +1 == size:
            print('maximum2')
            case = 3

            subject = sample[num]
            print('person', num+1, 'is', sample[num])

            leftNeighbour = sample[num - 2]
            righttNeighbour = 0
            print('The left neighbor of person', num, 'is', sample[num - 2])

            Left = True
            Right = False
            print('There is no right neighbour')

        else: # 0th case
            print('zeroth')
            case = 4

            subject = sample[num]
            print('person', num +1, 'is', sample[num])

            leftNeighbour = sample[num+1]
            rightNeighbour = 0
            print('The right neighbour of person', num+1, 'is', sample[num+1])

            Left = False
            Right = True
            print('There is no left neighbour')

    if size == 2:

        if num > 0:
            print('greater than 0')
            case = 5

            subject = sample[num]
            print('subject number is', num+1)
            print('person', num+1, 'is', sample[num], '\n')

            leftNeighbour = sample[num-1]
            righttNeighbour = 0
            print('The left neighbour of person', num, 'is', sample[num - 1])
            Left = True
            Right = False

        else:
            print('less than 0')
            case = 6

            subject = sample[num]
            print('subject number is', num+1)
            print('person', num+1, 'is', sample[num])

            lefttNeighbour = 0
            rightNeighbour = sample[num-1]
            print('The right neighbour of person', num, 'is', sample[num - 1])

            Left = False
            Right = True

    elif  size == 1:

        case = 7

        subject = sample[num]
        print('person is', sample[num], '\n')
        print('There is only one person in the sample')

        Left = False
        Right = False

    elif size == 0 or size > 100:

        print('corrupt smaple')

def select_random_neighbour(subject):

    s = random.randint(2)

    print('\nsubject =', subject)

    if population > 2:
        if Left == True and Right == True: #case 1

            if s == 0:
                opinion = subject - leftNeighbour
                print('left opinion =', opinion)
            elif s == 1:
                opinion = subject - rightNeighbour
                print('right opinion =', opinion)

        elif Left == True and Right == False: #case 2 & 3
            opinion = subject - leftNeighbour
            print('left opinion =', opinion)
        elif Left == False and Right == True: #case 4
            opinion = subject - rightNeighbour
            print('right opinion =', opinion)

    elif population == 2:

        if Left == True and Right == False: #case 5
            opinion = subject - leftNeighbour
            print('left opinion =', opinion)
        elif Left == False and Right == True: #case 6
            opinion = subject - rightNeighbour
            print('right opinion =', opinion)



    else:
        print('corrupt sample')

subject_and_neighbour(population, binary)
select_random_neighbour(subject)
