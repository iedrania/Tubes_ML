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
            output_array.append(output)
        return np.array(output_array)
