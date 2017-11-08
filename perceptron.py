"""
AI Lab: Perceptron
Author: Eduardo Vaca
"""
import random

class Perceptron(object):
    """ Class that represents a Perceptron node from ANN
    """

    def __init__(self, n_inputs, lower_limit=0.0001, upper_limit=0.01):
        self.threshold = random.uniform(lower_limit, upper_limit)
        self.weights = [random.uniform(lower_limit, upper_limit) for _ in range(n_inputs)]

    def __str__(self):
        return 'Threshold: {}\nWeights: {}'.format(self.threshold, self.weights)

    def train(self, x_train, y_train, learn_rate=0.005, iteration_limit=10):
        error = 1.0
        iteration = 0
        while error > 0.0 and iteration < iteration_limit:
            error = 0.0
            for i_record in range(len(x_train)):
                current_y = 1 if sum([x_train[i_record][i]*self.weights[i] for i in range(len(x_train[i_record]))]) >= self.threshold else 0
                error += abs(y_train[i_record]-current_y)
                self.re_weight(learn_rate, x_train[i_record], y_train[i_record], current_y)
                print(self.weights)
            iteration += 1
        return False if error > 0.0 else True

    def re_weight(self, learn_rate, x_data, y, y_obtained):
        for i in range(len(x_data)):
            weight_diff = learn_rate * (y-y_obtained) * x_data[i]
            self.weights[i] += weight_diff


def get_train_data(n_attributes, n_train):
    """Reads input and gets training data
    PARAMS:
    - n_attributes : number of attributes in a record
    - n_train : number of train records
    RETURNS:
    - tuple with (x train data, y train data)
    """
    x_train = []
    y_train = []
    for _ in range(n_train):
        current_record = input().replace(' ', '').split(',')
        x_train.append([float(x) for x in current_record[:n_attributes]])
        y_train.append(float(current_record[-1]))
    return (x_train, y_train)

def get_test_data(n_test):
    """Reads input and gets testing data
    PARAMS:
    - n_test : number of tests
    RETURNS:
    - list of test x data
    """
    return [[float(x) for x in input().replace(' ', '').split(',')] for _ in range(n_test)]

def main():
    """Main program
    """
    n_attributes = int(input())
    n_train = int(input())
    n_test = int(input())
    data = get_train_data(n_attributes, n_train)
    print(data)
    test = get_test_data(n_test)
    print(test)
    p = Perceptron(n_attributes)
    print(p)
    r_train = p.train(data[0], data[1])
    print(r_train)


if __name__ == '__main__':
    main()