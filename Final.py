import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import random


# Task 1
def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip its value
    Inputs:
        population (numpy array)
        row (int)
        col (int)
        external (float)
    '''
    n_rows, n_cols = population.shape
    Si = population[row, col]

    # Upper left corner
    if row == 0 and col == 0:
        N1 = population[row + 1, col]
        N2 = population[row, col + 1]
        neighbors_sum = N1 + N2
    # Upper right corner
    elif row == 0 and col == n_cols - 1:
        N1 = population[row, col - 1]
        N2 = population[row + 1, col]
        neighbors_sum = N1 + N2
    # Bottom left corner
    elif row == n_rows - 1 and col == 0:
        N1 = population[row - 1, col]
        N2 = population[row, col + 1]
        neighbors_sum = N1 + N2
    # Bottom right corner
    elif row == n_rows - 1 and col == n_cols - 1:
        N1 = population[row, col - 1]
        N2 = population[row - 1, col]
        neighbors_sum = N1 + N2
    # Upper edge
    elif row == 0 and 0 < col < n_cols - 1:
        N1 = population[row, col - 1]
        N2 = population[row, col + 1]
        N3 = population[row + 1, col]
        neighbors_sum = N1 + N2 + N3

    # Bottom edge
    elif row == n_rows - 1 and 0 < col < n_cols - 1:
        N1 = population[row - 1, col]
        N2 = population[row, col - 1]
        N3 = population[row, col + 1]
        neighbors_sum = N1 + N2 + N3

    # Left edge
    elif col == 0 and 0 < row < n_rows - 1:
        N1 = population[row, col + 1]
        N2 = population[row + 1, col]
        N3 = population[row - 1, col]
        neighbors_sum = N1 + N2 + N3

    # Right edge
    elif col == n_cols - 1 and 0 < row < n_rows - 1:
        N1 = population[row, col - 1]
        N2 = population[row + 1, col]
        N3 = population[row - 1, col]
        neighbors_sum = N1 + N2 + N3

    # Interior points
    else:
        N1 = population[row, col - 1]
        N2 = population[row, col + 1]
        N3 = population[row + 1, col]
        N4 = population[row - 1, col]
        neighbors_sum = N1 + N2 + N3 + N4

    Di = Si * (neighbors_sum + external)
    return Di

def ising_step(population, alpha=1.0, external=0.0):
    '''
    find the change_in_agreement (float)
    '''
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col, external)
    if agreement < 0 or np.random.random() < np.exp(-agreement / alpha):
        population[row, col] *= -1



def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.where(population == 1, 255, 1).astype(np.int8)
    im.set_data(new_im)
    plt.pause(0.1)



def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''


    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

    print("Tests passed")


def ising_main(population, external=0.0, alpha = 1, frames=100, steps=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(frames):
        # Iterating single steps 1000 times to form an update
        for step in range(steps):
            ising_step(population, alpha, external)
        plot_ising(im, population)
        print('Step:', frame, end='\r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Ising model simulation.")
    parser.add_argument("-ising_model", action="store_true", help="Activate the Ising model simulation.")
    parser.add_argument("-external", type=float, default=0.0, help="External influence factor.")
    parser.add_argument("-alpha", type=float, default=1.0, help="Parameter alpha.")
    parser.add_argument("-test_ising", action="store_true", help="Run test functions for the Ising model.")

    args = parser.parse_args()


    if args.test_ising:
        # To define what the test_ising should do
        test_ising()

    elif args.ising_model:
        population = np.random.choice([-1, 1], size = (100,100))
        ising_main(population, args.external, args.alpha, frames=100, steps=1000)

# Task 2

population = random.randint(25,100)  # population is ranged between 25 and 100

array = np.arange(population)

opinions = []
subjects = []
timestep = []

original = np.arange(population)  # the list that contains the original opinions
continuousScale = np.random.rand(population) # the updated list of original opinions with values in a continuous scale


for i in range(population): # appending values between 0 and 1 on a continuous scale to the 'continuous scale' list
    x = round(continuousScale[i], 3)
    continuousScale[i] = x
    i += 1

finalOpinions = continuousScale # creating a duplicate list of the original opinions where updated opinions will be appended to

def number_of_neighbours(size, sample, index):
    global case
    global subject
    global leftNeighbour
    global rightNeighbour
    global neighbour

    neighbourNum = 1  # the index of the neighbour

    if size > 2 and size < 100:  # checking if the population is less than 100 and greater than 2
        if index > 0 and (index + 1) != size:  # middle - subject not on the edge, 2 neighbours
            #subject = sample[index] # looping through the sample to select every element as the subject
            rightNeighbour = sample[index + 1]
            leftNeighbour = sample[index - 1]
            case = 1

        elif index == size:  # maximum case - subject is at the end of the list, 1 neighbour
            #subject = sample[index - 1]
            leftNeighbour = sample[index - 2] # the subject only has a left neighbour
            rightNeighbour = 0
            case = 2

        elif index + 1 == size:  # looping to the last element (opinion), 1 neighbour
            #subject = sample[index]  # the index of the subject is 'num' since the index of the last element is the list size - 1
            leftNeighbour = sample[index - 2] # the subject only has a left neighbour
            rightNeighbour = 0
            case = 3

        else:  # 0th case - this is the first element (opinion) of the list, 1 neighbour
            #subject = sample[index]
            leftNeighbour = 0
            rightNeighbour = sample[index + 1] # the subject only has a right neighbour
            case = 4

    elif size == 2: # checking if the population is 2, 1 neighbour
        if index > 0:  # this subject is at the end of the list
            #subject = sample[index]
            leftNeighbour = sample[index - 1] # the subject only has a left neighbour
            rightNeighbour = 0
            case = 5

        else:  # this subject is at the beginning of the list
            #subject = sample[index]
            leftNeighbour = 0
            rightNeighbour = sample[index - 1] # the subject only has a right neighbour
            case = 6

    else :  # checking if the population is 1, 0 , or out of range
        case = 0

def selectingNeighbours(size, sample, threshold, beta):
    global neighbourNum
    global neighbour
    global opinion
    global subject
    global num
    global s

    subject = 0
    num = 0 # index of the subject
    neighbourNum = 0 # the index of the neighbour

    fig, ax = plt.subplots(1, 2)

    for num in range(size):
        number_of_neighbours(size, sample, num)
        subject = sample[num]
        s = random.randint(2)  # variable with two random possible values for selecting the left or right neighbour

        if case == 1: # checking if the subject has both a left and right neighbour
            if s == 0: # left neighbour is selected
                neighbourNum = num - 1  # the left neighbour's index is 1 less than the subject's index
                neighbour = leftNeighbour
                opinion = subject - neighbour  # the opinion is the difference of the subject and a random neighbour
            else:
                neighbourNum = num + 1  # # the right neighbour's index is 1 greater than the subject's index
                neighbour = rightNeighbour
                opinion = subject - neighbour

        elif case == 2 or case ==3 or case == 5: # checking if the subject only has a left neighbour
            neighbourNum = num - 1
            neighbour = leftNeighbour
            opinion = subject - neighbour

        elif case == 4 or case == 6: # checking if the subject only has a right neighbour
            neighbourNum = num + 1
            neighbour = rightNeighbour
            opinion = subject - neighbour

        opinion = round(opinion, 3)

        for i in range(size):
            ax[1].scatter(array[num], finalOpinions[i], color = 'red')
            i += 1

        updateOpinion(opinion, sample, threshold, beta)

        num += 1

    for i in range(size):
        ax[1].scatter(array[num-1], finalOpinions[i], color = 'red') # ploting the final values
        i += 1
    ax[0].hist(finalOpinions)

    ax[0].set_xlabel('opinions')
    ax[1].set_ylabel('opinions')
    ax[0].set_title('Coupling:' + str(beta))
    ax[1].set_title('Threshold:' + str(threshold))

    plt.show()
def updateOpinion(value, sample, threshold, beta):
    global subject2
    global neighbour2

    subject2 = 0
    neighbour2 = 0

    #print('num', num)
    if abs(value) > threshold:

        subject2 = round(sample[num] + beta * (neighbour - sample[num]), 3)
        finalOpinions[num] = subject2

        neighbour2 = round(finalOpinions[num] + beta * (finalOpinions[num] - neighbour), 3)
        finalOpinions[neighbourNum] = neighbour2

    else:
        subject2 = sample[num]
        neighbour2 = neighbour

    finalOpinions[num] = subject2
    finalOpinions[neighbourNum] = neighbour2

def defaunt_main(threshold = 0.2, beta = 0.2):
    global T
    global Beta
    global subject2
    global neighbour2

    selectingNeighbours(population, continuousScale, threshold, beta)
    print('completed')

def defaunt_test():
    print(population)
    selectingNeighbours(population)
    #print(continuousScale)
    #print(opinions)

    print('1new sub:', finalOpinions[num])
    print('2new sub:', finalOpinions[num])

    print(num, 's', subject, 'to', subject2)
    print(sample[num], '+', beta, '*', '(', neighbour, '-', sample[num], ')')
    print(num, 'n', neighbour, 'to', neighbour2)
    print(finalOpinions[num], '+', beta, '*', '(', finalOpinions[num], '-', neighbour, ')')
    print(num, 'changed \n')

defaunt_main()



#Task 3

class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

class Network: 

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

   
    # calculate the mean degree of the network
       def get_mean_degree(self):
        total_degree = 0
        # Get the number of nodes in the network
        num_nodes = len(self.nodes)

        for node in self.nodes:
            # Sum up the connections of each node
            total_degree += sum(node.connections)

        if num_nodes > 0:
            return total_degree / num_nodes
        else:
            return 0 # Return 0 if there are no nodes in the network

    # Calculate the clustering coefficient of the network
    def clustering_coefficient(self):
        total_coefficient = 0
        # Get the number of nodes in the network
        num_nodes = len(self.nodes)

        for node in self.nodes:
            num_neighbours = sum(node.connections)
            if num_neighbours > 1:
                num_possible_connections = (num_neighbours * (num_neighbours - 1)) / 2
                num_actual_connections = 0
                # Check actual connections between neighbors
                neighbour_indices = [i for i, conn in enumerate(node.connections) if conn]
                for i, neighbour_index in enumerate(neighbour_indices):
                    for j in range(i + 1, len(neighbour_indices)):
                        neighbour_index_2 = neighbour_indices[j]
                        if self.nodes[neighbour_index_2].connections[neighbour_index]:
                            num_actual_connections += 1

                if num_possible_connections > 0:
                    coefficient = num_actual_connections / num_possible_connections
                    total_coefficient += coefficient

        if num_nodes > 0:
            return total_coefficient / num_nodes
        else:
            return 0

    # Calculate and the mean path length of the network
    def mean_path_length(self):
        total_path_length = 0
        num_pairs = 0

        num_nodes = len(self.nodes)
        # Iterate through each pair of nodes in the network
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Find the shortest path length between the pair of nodes
                    path_length = self.bfs_shortest_path(i, j)
                    total_path_length += path_length
                    num_pairs += 1

        if num_pairs > 0:
            return total_path_length / num_pairs
        else:
            return 0

    # Implement breadth-first search to find the shortest path between two nodes
    def bfs_shortest_path(self, start_node, end_node):
        visited = set()
        queue = [(start_node, 0)]

        while queue:
            node, distance = queue.pop(0)
            if node == end_node:
                return distance
            visited.add(node)
            # Explore neighbors of the current node
            for neighbour_index, connected in enumerate(self.nodes[node].connections):
                if connected and neighbour_index not in visited:
                    queue.append((neighbour_index, distance + 1))

        return float('inf')
        

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1



    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


def test_networks():

    #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.clustering_coefficient()==0), network.clustering_coefficient()
    assert round(network.mean_path_length(), 15) == 2.777777777777778, network.mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.clustering_coefficient()==0),  network.clustering_coefficient()
    assert(network.mean_path_length()==5), network.mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.clustering_coefficient()==1),  network.clustering_coefficient()
    assert(network.mean_path_length()==1), network.mean_path_length()

    print("All tests passed")



def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    

    return np.random.random() * population

def ising_step(population, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col  = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external=0.0)

    if agreement < 0:
        population[row, col] *= -1

    #Your code for task 1 goes here

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Network Analysis')
    parser.add_argument('-network', type=int, help='Create and plot a random network of specified size')
    parser.add_argument('-test_network', action='store_true', help='Run test functions')

    args = parser.parse_args()

    if args.network:
        network = Network()
        network.make_random_network(args.network)
        print('mean degree', network.get_mean_degree())
        print('mean path length', network.mean_path_length())
        print('clustering_coefficient', network.clustering_coefficient())
        network.plot()
        plt.show()

    if args.test_network:
        test_networks()

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




