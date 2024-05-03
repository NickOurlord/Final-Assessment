import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


--------------------------------------------------------------------------------------------------------------------------------------------------------------
#Task 1 Code
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
--------------------------------------------------------------------------------------------------------------------------------------------------------------

#Task 3 Code

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

    # Define arguments of the command lines

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
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1



    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


def test_networks():
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.clustering_coefficient() == 0), network.clustering_coefficient()
    assert round(network.mean_path_length(), 15) == 2.777777777777778, network.mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.clustering_coefficient() == 0), network.clustering_coefficient()
    assert (network.mean_path_length() == 5), network.mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.clustering_coefficient() == 1), network.clustering_coefficient()
    assert (network.mean_path_length() == 1), network.mean_path_length()

    print("All tests passed")
--------------------------------------------------------------------------------------------------------------------------------------------------------------












--------------------------------------------------------------------------------------------------------------------------------------------------------------
Flags go below here
--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Ising model simulation.")
    parser.add_argument("-ising_model", action="store_true", help="Activate the Ising model simulation.")
    parser.add_argument("-external", type=float, default=0.0, help="External influence factor.")
    parser.add_argument("-alpha", type=float, default=1.0, help="Parameter alpha.")
    parser.add_argument("-test_ising", action="store_true", help="Run test functions for the Ising model.")
    # Task3
    parser = argparse.ArgumentParser(description='Network Analysis')
    parser.add_argument('-network', type=int, help='Create and plot a random network of specified size')
    parser.add_argument('-test_network', action='store_true', help='Run test functions')








  
    args = parser.parse_args()




--------------------------------------------------------------------------------------------------------------------------------------------------------------
Flag if statements go below
--------------------------------------------------------------------------------------------------------------------------------------------------------------

    if args.test_ising:
        # To define what the test_ising should do
        test_ising()

    elif args.ising_model:
        population = np.random.choice([-1, 1], size = (100,100))
        ising_main(population, args.external, args.alpha, frames=100, steps=1000)


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




      
