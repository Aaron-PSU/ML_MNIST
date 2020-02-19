# Aaron Dawson
# PSUID: 900818217
# 2/10/20
# Hidden layer implementation of ML project 1
# Classifies images from the MNIST set as digits 0-9

import numpy as np
import matplotlib.pyplot as plt
e = 2.718281828459


class Network(object):
    def __init__(self, training_data_received, layers):
        self.master_training_data = training_data_received
        self.num_layers = len(layers)
        self.layers = layers
        print("number of layers: {0}\nlayers: {1}".format(self.num_layers, self.layers))
        # need to first figure out the size/shape of weights
        # then I need to fill with random weights.
        self.input_to_hidden_weights = np.random.uniform(low=-0.05, high=0.05, size=[self.layers[1], self.layers[0]+1])
        self.hidden_to_output_weights = np.random.uniform(low=-0.05, high=0.05, size=[self.layers[2], self.layers[1]+1])
        self.hidden_activations = np.zeros(self.layers[1]+1)
        self.confusion_matrix = np.zeros([10, 10])

    def set_i2h_weight(self, row, column, new_weight):
        self.input_to_hidden_weights[row, column] = new_weight

    def set_h2o_weight(self, row, column, new_weight):
        self.hidden_to_output_weights[row, column] = new_weight

    def print_weight_matrices(self):
        print("\ninput to hidden weights:\n{0}\n"
              "\nhidden to output weights:\n{1}".format(self.input_to_hidden_weights,
                                                        self.hidden_to_output_weights))

    # forward propagation
    # input  : one row from master training data / testing data where a[0] is target answer
    # returns: output_activations array
    def forward_prop(self, input_array):
        # create hidden activations array of the correct size and initialize inputs array from Master_training_data
        self.hidden_activations[0] = 1
        inputs = np.array(input_array)
        inputs[0] = 1

        i2h_dot_product = np.dot(self.input_to_hidden_weights, inputs)
        mapped_hidden_activations = map(sigmoid, i2h_dot_product)
        self.hidden_activations[1:] = list(mapped_hidden_activations)

        h2o_dot_product = np.dot(self.hidden_to_output_weights, self.hidden_activations)
        output_activations = map(sigmoid, h2o_dot_product)

        return list(output_activations)

    # reliant on forward propagation to have already occurred, so that hidden activations are current
    def back_prop(self, input_array, learning_rate, output_activations):
        # calculate error for each OUTPUT unit following equations from lecture
        target_array = create_target_array(self.layers[2], input_array[0])
        output_error_matrix = target_array - output_activations
        output_error_matrix *= output_activations
        one_minus_output_activations = np.ones(len(output_activations))
        one_minus_output_activations -= output_activations
        output_error_matrix *= one_minus_output_activations

        # first reorient the hidden_to_output_weights for easier matrix multiplication
        flipped_weights = np.array(self.hidden_to_output_weights[:, 1:])
        flipped_weights = np.rot90(flipped_weights, 3)
        flipped_weights = np.flip(flipped_weights, 1)

        # calculate error for each HIDDEN unit following equations from lecture
        hidden_layer_error_matrix = np.dot(flipped_weights, output_error_matrix)
        # hidden_layer_error_matrix = np.dot(self.hidden_to_output_weights[:, 1:], output_error_matrix)
        hidden_layer_error_matrix *= self.hidden_activations[1:]
        one_minus_hidden_activations = np.ones(len(self.hidden_activations[1:]))
        one_minus_hidden_activations -= self.hidden_activations[1:]
        hidden_layer_error_matrix *= one_minus_hidden_activations

        # change h2o weights
        h2o_delta = np.ones(self.hidden_to_output_weights.shape)
        h2o_delta *= self.hidden_activations
        h2o_delta *= learning_rate
        turned_matrix = np.ones([len(output_error_matrix), 1])
        for i in range(len(output_error_matrix)):
            turned_matrix[i] = output_error_matrix[i]
        h2o_delta *= turned_matrix
        self.hidden_to_output_weights += h2o_delta

        # change i2h weights
        i2h_delta = np.ones(self.input_to_hidden_weights.shape)
        input_array_with_bias = np.array(input_array)
        input_array_with_bias[0] = 1
        i2h_delta *= input_array_with_bias
        i2h_delta *= learning_rate
        turned_matrix = np.ones([len(hidden_layer_error_matrix), 1])
        for i in range(len(hidden_layer_error_matrix)):
            turned_matrix[i] = hidden_layer_error_matrix[i]
        i2h_delta *= turned_matrix
        self.input_to_hidden_weights += i2h_delta

    # trains the network and plots the accuracy graph
    def train(self, epochs, learning_rate, test_data=None):
        accuracy_array_train = np.zeros(epochs+1)
        accuracy_array_test = np.zeros(epochs+1)
        accuracy_array_train[0] = self.test_accuracy(self.master_training_data)
        accuracy_array_test[0] = self.test_accuracy(test_data)

        # for the number of epochs
        for j in range(epochs):
            print("start of epoch {0}".format(j+1))
            np.random.shuffle(self.master_training_data)

            # iterate through each example in the training data
            # NOTE: for part two of homework, this for loop was adjusted
            for k in range(len(self.master_training_data)):
                self.back_prop(self.master_training_data[k], learning_rate,
                               my_network.forward_prop(self.master_training_data[k]))

            # fill accuracy arrays with the number of examples they got correct
            accuracy_array_train[j+1] = self.test_accuracy(self.master_training_data)
            accuracy_array_test[j+1] = self.test_accuracy(test_data)

        # after all epochs, divide by the length of the data set to get the accuracy
        accuracy_array_train /= len(self.master_training_data)
        accuracy_array_test /= len(test_data)

        # plot the accuracies in a nice graph
        plt.plot(accuracy_array_test,  label='testing data')
        plt.plot(accuracy_array_train,  label='training data')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title("Accuracy for Network {0}".format(learning_rate))
        plt.legend()
        plt.show()

    def best_guess(self, x):
        guess = np.argmax(self.forward_prop(x))
        return guess

    def test_accuracy(self, data_to_test):
        counter = 0
        for i in range(len(data_to_test)):
            guess = self.best_guess(data_to_test[i])
            answer = data_to_test[i, 0]
            if guess == answer:
                counter += 1
        return counter
    # capture the perceptrons answers to the test data

    def fill_confusion_matrix(self, data_to_test):
        self.confusion_matrix = np.zeros([10, 10])
        for i in range(len(data_to_test)):
            guess = self.best_guess(data_to_test[i])
            answer = data_to_test[i, 0]
            self.confusion_matrix[int(answer), int(guess)] += 1

    def print_confusion_matrix(self):
        np.set_printoptions(suppress=True)
        print(self.confusion_matrix)


# returns a target vector of "0.1's"  with the "j"th answer as a 0.9
def create_target_array(length, j):
    out = np.zeros(length)
    out += 0.1
    out[int(j)] += 0.8
    return out


def sigmoid(x):
    return 1/(1 + (pow(e, -x)))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# load the .csv file into a numpy array
def load_mnist(file):
    arrays = np.loadtxt(file, delimiter=',')
    preprocessed_arrays = preprocess_mnist(arrays)
    return preprocessed_arrays


def preprocess_mnist(array):
    new_array = array / 255
    for x in range(len(new_array)):
        # leave first element unchanged in each row, signifying the correct answer
        new_array[x, 0] *= 255
    return new_array


# reduce master data by dividing by div amount whilst insuring that all 10 numbers stay reasonably balanced
def div_master_data(array, div):

    # first, separate each input into a separation_array where its[x,,] value
    # corresponds with that input's correct answer (10 possible answers 0-9)
    separation_array = np.zeros([10, int(len(array)/5), int(len(array[0]))])
    # "place" array stores each containers' next index to be filled (10 containers 0-9)
    place = np.zeros(shape=10, dtype=int)
    index_to_sort = 0
    np.random.shuffle(array)
    while index_to_sort < len(array) and (np.argmin(place) * 10) < (len(array)/div):
        for k in range(10):
            if array[index_to_sort, 0] == k:
                separation_array[k, place[k]] = array[index_to_sort]
                # print("{0} sorted to {1}".format(array[index_to_sort, 0], k))
                place[k] += 1
        index_to_sort += 1

    # second, fill an array with a balanced set of training data
    new_array = np.zeros([int(len(array)/div), int(len(array[0]))])
    place = np.zeros(shape=10, dtype=int)
    current_answer_type_to_add = 0
    for i in range(int(len(array)/div)):
        new_array[i] = separation_array[current_answer_type_to_add, place[current_answer_type_to_add]]
        place[current_answer_type_to_add] += 1
        current_answer_type_to_add += 1
        if current_answer_type_to_add == 10:
            current_answer_type_to_add = 0
            # print("current_answer_type_to_add is 10! resetting it to 0")
        # print(i)

    # NOTE: new_array answers follow pattern 0, 1, 2, ..... 8, 9, 0, 1, 2,... etc...
    # since train() shuffles it's training data before each epoch, this won't be an issue
    return new_array


if __name__ == '__main__':
    training_data = load_mnist("mnist_train.csv")
    # training_data = load_mnist("mnist_first500.csv")
    testing_data = load_mnist("mnist_test.csv")

    # training_data_half = div_master_data(training_data, 2)
    # training_data_quartered = div_master_data(training_data_half, 2)
    # training_data_div_120 = div_master_data(training_data, 120)

    print("done loading the data")

    """
    # these weights come from the worked example from class.
    # used to test algorithm correctness
    
    training_data = [0, 1, 0]
    my_network = Network(training_data, [2, 2, 2])

    my_network.set_i2h_weight(0, 0, -0.4)
    my_network.set_i2h_weight(0, 1,  0.2)
    my_network.set_i2h_weight(0, 2,  0.1)
    my_network.set_i2h_weight(1, 0, -0.2)
    my_network.set_i2h_weight(1, 1,  0.4)
    my_network.set_i2h_weight(1, 2, -0.1)

    my_network.set_h2o_weight(0, 0,  0.1)
    my_network.set_h2o_weight(0, 1, -0.2)
    my_network.set_h2o_weight(0, 2,  0.1)
    my_network.set_h2o_weight(1, 0,  0.4)
    my_network.set_h2o_weight(1, 1, -0.1)
    my_network.set_h2o_weight(1, 2,  0.1)

    my_network.print_weight_matrices()
    print(my_network.back_prop(training_data, 0.1, my_network.forward_prop(training_data)))
    """

    # for experiments 1-3, differing amounts of hidden nodes
    # experiments 5 and 6, differing sizes of training data
    # *******************************************************************************************************
    # my_network = Network(training_data, [784, 20, 10])
    # my_network = Network(training_data, [784, 50, 10])
    my_network = Network(training_data, [784, 100, 10])
    # my_network = Network(training_data_half, [784, 100, 10])
    # my_network = Network(training_data_div_120, [784, 100, 10])

    # for these experiments train on 50 epochs using learning rate 0.1, and testing data from mnist_test.csv
    my_network.train(50, 0.1, testing_data)
    # *******************************************************************************************************

    print("\ntesting")
    my_network.fill_confusion_matrix(testing_data)
    my_network.print_confusion_matrix()

    print("\ntraining")
    my_network.fill_confusion_matrix(training_data)
    # my_network.fill_confusion_matrix(training_data_half)
    # my_network.fill_confusion_matrix(training_data_quartered)
    # my_network.fill_confusion_matrix(training_data_div_120)

    my_network.print_confusion_matrix()
