import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import random

population = random.randint(50)
#population = 2

print('population is', population, '\n')

original = np.arange(population)
binary = np.arange(population)

opinions = []

print('original sample: ')
print(original, '\n')

for i in range(population):
#    print(sample[i])
    x = random.randint(2)
    binary[i] = x
    i += 1
print('binary sample ')
print(binary, '\n')

global num
def subject_and_neighbour(size, sample):

#    num = size
    global subject
    global leftNeighbour
    global rightNeighbour
    global Left
    global Right
    global case

    num = 0

    for i in range(size):

        #print('num =', num)


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

        num +=1

        select_random_neighbour(subject)
        opinions.append(opinion)


def select_random_neighbour(subject):
    global opinion

    s = random.randint(2)

    if population > 2:
        if Left == True and Right == True: #case 1

            if s == 0:
                opinion = subject - leftNeighbour

            elif s == 1:
                opinion = subject - rightNeighbour

        elif Left == True and Right == False: #case 2 & 3
            opinion = subject - leftNeighbour

        elif Left == False and Right == True: #case 4
            opinion = subject - rightNeighbour

    elif population == 2:

        if Left == True and Right == False: #case 5
            opinion = subject - leftNeighbour

        elif Left == False and Right == True: #case 6
            opinion = subject - rightNeighbour

    else:
        opinion = 0
        print('corrupt sample')


subject_and_neighbour(population, binary)

print('opinions :\n', opinions)