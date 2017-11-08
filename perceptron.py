"""
AI Lab: Perceptron
Author: Eduardo Vaca
"""


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
        x_train.append([int(x) for x in current_record[:n_attributes]])
        y_train.append(int(current_record[-1]))
    return (x_train, y_train)

def get_test_data(n_test):
    """Reads input and gets testing data
    PARAMS:
    - n_test : number of tests
    RETURNS:
    - list of test x data
    """
    return [[int(x) for x in input().replace(' ', '').split(',')] for _ in range(n_test)]

def main():
    """Main program
    """
    n_attributes = int(input())
    n_train = int(input())
    n_test = int(input())
    result = get_train_data(n_attributes, n_train)
    print(result)
    test = get_test_data(n_test)
    print(test)


if __name__ == '__main__':
    main()