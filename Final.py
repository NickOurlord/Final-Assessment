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


#Task 4

# Create an argument parser to parse command line arguments
parser = argparse.ArgumentParser()

# Define arguments of the command lines
parser.add_argument('-ring_network', nargs='?', const=10, type=int,
                    help='Generate a ring network of specified size (default: 10)')
parser.add_argument('-small_world', nargs='?', const=10, type=int,
                    help='Generate a small world network of specified size (default: 10)')
parser.add_argument('-re_wire', default=0.2, type=float,
                    help='Rewire probability for small world network (default: 0.2)')

# Parse the command line arguments
args = parser.parse_args()

# Define a class representing a Node in the network
class Node:

    def __init__(self, value, number, connections=None):

        self.index = number  # Node index
        self.connections = connections  # Node connections
        self.value = value  # Node value

# Define a class to represent a Network
class Network: 

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []  # Initialize node array
        else:
            self.nodes = nodes 

    # Function to create a random network of size N with connection probability p
    def make_random_network(self, N, connection_probability):
        '''
        This function creates a random network of size N, 
        where each node is connected to every other node with a given probability p.

        '''

        self.nodes = []  # Initialize node array
        for node_number in range(N):
            value = np.random.random()  # Random value for the node
            connections = [0 for _ in range(N)]  # Initialize connections
            self.nodes.append(Node(value, node_number, connections))

        # Connect nodes with probability p
        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    # Function to create a ring network of size N with specified neighbour range
    def make_ring_network(self, N, neighbour_range=1):
        '''
        This function generates a ring network of size N with the specified neighbour range.
        '''
        self.nodes = []  # Initialize node array
        for node_number in range(N):
            value = np.random.random()  # Random value for the node
            connectivity = np.zeros(N, dtype=int)  # Initialize connectivity array
            for i in range(N):
                if abs(i - node_number) <= neighbour_range and abs(i - node_number) != 0 or abs(i - node_number) >= N - neighbour_range:
                    connectivity[i] = 1  # Connect nodes within the specified range
            self.nodes.append(Node(value, node_number, connectivity))
            
    # Function to create a small-world network of size N with specified rewire probability
    def make_small_world_network(self, N, re_wire_prob=0.2):
        '''
        This function generates a small-world network of size N with the specified rewire probability.
        '''
        self.nodes = []  # Initialize node array
        for node_number in range(N):
            value = np.random.random()  # Random value for the node between 0 and 1
            connectivity = np.zeros(N, dtype=int)  # Initialize connectivity array
            for i in range(N):
                if abs(i - node_number) <= 2 and abs(i - node_number) != 0 or abs(i - node_number) >= N - 2:
                    connectivity[i] = 1  # Connect nodes within the specified range
            self.nodes.append(Node(value, node_number, connectivity))
            
        # Rewire edges with a certain probability
        for node in self.nodes:
            temp_connections = np.copy(node.connections)
            for i, connection in enumerate(node.connections):
                if connection == 1 and np.random.random() < re_wire_prob:
                    temp_connections[i] = 0
                    while True:
                        rand_node = np.random.randint(0, N-1)
                        if temp_connections[rand_node] == 0:
                            temp_connections[rand_node] = 1
                            break
            node.connections = np.copy(temp_connections)

    # Function to plot the network
    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)  # Number of nodes in the network
        network_radius = num_nodes * 10  # Radius of the network visualization
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])  # Set x-axis limits
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])  # Set y-axis limits

        # Plot nodes and edges
        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes  # Angle of the node in the network
            node_x = network_radius * np.cos(node_angle)  # x-coordinate of the node
            node_y = network_radius * np.sin(node_angle)  # y-coordinate of the node

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))  # Circle representing the node
            ax.add_patch(circle)

            # Plot edges to neighboring nodes
            for neighbour_index in range(0, num_nodes): 
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes  # Angle of the neighboring node
                    neighbour_x = network_radius * np.cos(neighbour_angle)  # x-coordinate of the neighboring node
                    neighbour_y = network_radius * np.sin(neighbour_angle)  # y-coordinate of the neighboring node

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')  # Plot edge between nodes

        plt.show()  # Display the plot

# Calling the function
if __name__ == '__main__':
    network = Network()  

    # Check command line arguments and generate corresponding network
    if args.ring_network:
        ring_size = args.ring_network
        print(f'Creating a size {ring_size} ring network')
        network.make_ring_network(ring_size)
        network.plot()

    elif args.small_world:
        if not 0 <= args.re_wire <= 1:
            print('Error')
            quit()

        world_size = args.small_world
        re_wire_prob = args.re_wire
        print(f'Creating a size {world_size} small world network with re-wire probability of {re_wire_prob}')
        network.make_small_world_network(world_size, re_wire_prob)
        network.plot()




