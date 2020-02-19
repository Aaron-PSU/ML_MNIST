# Aaron Dawson
# PSUID: 900818217

import numpy as np
# import matplotlib.pyplot as plt
e = 2.718281828459


class Network(object):
    def __init__(self, training_data_received, layers):
        self.master_training_data = training_data_received
        self.num_layers = len(layers)
        self.layers = layers
        print("number of layers: {0}\nlayers: {1}".format(self.num_layers, self.layers))
        # need to first figure out the size/shape of weights
        # then I need to fill with random weights.
        self.input_to_hidden_weights = np.zeros(shape=[self.layers[1], (self.layers[0]+1)])
        self.hidden_to_output_weights = np.zeros(shape=[self.layers[2], (self.layers[1]+1)])

        # [np.random.uniform(low=-0.05, high=0.05, size=[l0, 785])
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
    def forward_prop(self, a):
        # create hidden activations array of the correct size and initialize inputs array from Master_training_data
        hidden_activations = np.zeros(self.layers[1]+1)
        hidden_activations[0] = 1
        inputs = np.array(a)
        inputs[0] = 1

        i2h_dot_product = np.dot(self.input_to_hidden_weights, inputs)
        mapped_hidden_activations = map(sigmoid, i2h_dot_product)
        hidden_activations[1:] = list(mapped_hidden_activations)

        h2o_dot_product = np.dot(self.hidden_to_output_weights, hidden_activations)
        output_activations = map(sigmoid, h2o_dot_product)

        return list(output_activations)


"""
    # adjusts all weights in accordance too the perceptron learning rule
    # wi <-- wi + n(t-y)xi
    def adjust_all_weights(self, learning_rate, t, y, x):
        inputs = self.master_training_data[x]
        shape = self.weights.shape
        pos_or_neg_array = t-y
        w = np.ones(shape)
        w *= learning_rate
        w *= inputs
        for i in range(len(w)):
            w[i] *= pos_or_neg_array[i]
        self.weights = np.add(self.weights, w)

    # trains the network and plots the accuracy graph
    def train(self, epochs, learning_rate, test_data=None):
        accuracy_array_train = np.zeros(epochs+1)
        accuracy_array_test = np.zeros(epochs+1)
        accuracy_array_train[0] = self.test_training(self.master_training_data)
        accuracy_array_test[0] = self.test_training(test_data)

        # for the number of epochs
        for j in range(epochs):
            print("start of epoch {0}".format(j+1))
            np.random.shuffle(self.master_training_data)

            # iterate through each example in the training data
            for k in range(len(self.master_training_data)):
                current_true_answer = self.master_training_data[k, 0]
                answer_key = result_as_array(int(current_true_answer))
                perceptron_answers = self.calculate_output(self.master_training_data[k])

                # convert perceptron answers from float form into binary form
                # for use with the function "adjust_all_weights"
                for m in range(10):
                    if perceptron_answers[m] > 0:
                        perceptron_answers[m] = 1
                    else:
                        perceptron_answers[m] = 0

                self.adjust_all_weights(learning_rate, answer_key, perceptron_answers, k)

            # fill accuracy arrays with the number of examples they got correct
            accuracy_array_train[j+1] = self.test_training(self.master_training_data)
            accuracy_array_test[j+1] = self.test_training(test_data)

        # after all epochs, divide by the length of the data set to get the accuracy
        accuracy_array_train /= len(self.master_training_data)
        accuracy_array_test /= len(test_data)

        # plot the accuracies in a nice graph
        plt.plot(accuracy_array_test,  label='testing data')
        plt.plot(accuracy_array_train,  label='training data')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title("Accuracy for learning rate {0}".format(learning_rate))
        plt.legend()
        plt.show()

    def best_guess(self, x):
        #guess = np.argmax(self.calculate_output(x))
        #return guess

    def test_training(self, data_to_test):
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
"""


# returns a vector of 10 zeros with the "j"th answer as a 1.0
def result_as_array(j):
    out = np.zeros(10)
    out[j] = 1.0
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
        # fix that first element in each row signifying the correct answer
        new_array[x, 0] *= 255
    return new_array


if __name__ == '__main__':
    # training_data = load_mnist("mnist_train.csv")
    # testing_data = load_mnist("mnist_test.csv")
    # print("done loading the data")

    training_data = [1, 1, 0]
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
    print(my_network.forward_prop(training_data))

    # *******************************************************************************************
#    my_network.train(50, 0.1, testing_data)
    # *******************************************************************************************

    # print("\ntesting")
    # my_network.fill_confusion_matrix(testing_data)
    # my_network.print_confusion_matrix()

    # print("\ntraining")
    # my_network.fill_confusion_matrix(training_data)
    # my_network.print_confusion_matrix()
