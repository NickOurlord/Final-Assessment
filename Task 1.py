import numpy as np
import matplotlib.pyplot as plt
import argparse


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
