from black import err
import numpy as np
from perceptron import Perceptron

class Layer:
    def __init__(self, layer_type, actifunc, arr_perceptron):
        self.type = layer_type # output or hidden
        self.activation_function_type = actifunc
        self.array_of_perceptron = arr_perceptron

    def calculate_all(self, arr_input):
        output_array = []
        # for perceptron in self.array_of_perceptron:
        for i in range (len(self.array_of_perceptron)):
            perceptron = self.array_of_perceptron[i]
            output = perceptron.calculate(arr_input, self.activation_function_type)
            # print("ini output di calculate all")
            output_array.append(output)
        return np.array(output_array)

    def calculate_back_hidden(self, learning_rate, arr_error_before, arr_weights_before):
        arr_error = np.empty((0, len(arr_error_before[0])), float)
        arr_weights = np.empty((0, len(self.array_of_perceptron[0].weights)), float)
        for i in range (len(self.array_of_perceptron)):
            perceptron = self.array_of_perceptron[i]
            weights = np.array([])
            for j in range (len(arr_weights_before)):
                weights = np.append(weights, arr_weights_before[j][i])
            arr_weights = np.append(arr_weights, [perceptron.weights], axis = 0)
            perceptron.update_weight_hidden(learning_rate, self.activation_function_type, arr_error_before, weights)
            arr_error = np.append(arr_error, [perceptron.errors], axis = 0)
        return arr_error, arr_weights

    def calculate_back_output(self,learning_rate, target):
        arr_error = np.empty((0, len(target)), float)
        arr_weights = np.empty((0, len(self.array_of_perceptron[0].weights)), float)
        if (self.activation_function_type == "softmax"):
            for i in range (len(self.array_of_perceptron)):
                perceptron = self.array_of_perceptron[i]   
                arr_weights = np.append(arr_weights, [perceptron.weights], axis = 0)
                perceptron.update_weight_output_softmax(learning_rate, target)
                arr_error = np.append(arr_error, [perceptron.errors], axis = 0)
        else:
            for i in range (len(self.array_of_perceptron)):
                perceptron = self.array_of_perceptron[i]
                arr_weights = np.append(arr_weights, [perceptron.weights], axis = 0)
                perceptron.update_weight_output(learning_rate, self.activation_function_type, target)
                arr_error = np.append(arr_error, [perceptron.errors], axis = 0)
        return arr_error, arr_weights