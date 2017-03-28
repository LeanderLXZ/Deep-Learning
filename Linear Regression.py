from numpy import *
from matplotlib import pyplot as plt


def error_of_line(b, m, data):
    error = 0
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        error += (y - (m * x + b))**2
    return error / float(len(data))


def plot_show(b, m, data):
    x = [c[0] for c in data]
    y = [d[1] for d in data]
    z = [i * m + b for i in x]
    plt.plot(x, z, 'b-', x, y, 'ro')
    plt.show()


def gradient(current_b, current_m, data):
    b_grad = 0
    m_grad = 0
    N = float(len(data))
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        b_grad += - (2 / N) * (y - (current_m * x + current_b))
        m_grad += - (2 / N) * x * (y - (current_m * x + current_b))
    return [b_grad, m_grad]


def updating(start_b, start_m, learningRate, iterationNum, point):
    updated_b = start_b
    updated_m = start_m
    for i in range(iterationNum):
        [gradient_b, gradient_m] = gradient(updated_b, updated_m, array(point))
        updated_b += - learningRate * gradient_b
        updated_m += - learningRate * gradient_m
        if (i < 10) or (i % 100 == 0):
            print("> Updating... >  iteration: {0} > b = {1}, m = {2}, error = {3}".format(i, updated_b, updated_m, error_of_line(updated_b, updated_m, point)))
            plot_show(updated_b, updated_m, point)
    return [updated_b, updated_m]


def run():
    data = genfromtxt('data.csv', delimiter=',')
    initial_b = 0
    initial_m = 0
    learning_rate = 0.0001
    iteration_num = 1000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, error_of_line(initial_b, initial_m, data)))
    print("Running...")
    [b, m] = updating(initial_b, initial_m, learning_rate, iteration_num, data)
    print("> ........... >")
    print("Afeter {0} iterations b = {1}, m = {2}, error = {3}".format(iteration_num, b, m, error_of_line(b, m, data)))

    plot_show(b, m, data)

if __name__ == '__main__':
    run()