from distutils.log import error
import errno
from black import err
import numpy as np
# from torch import sigmoid

class Perceptron:

    def __init__(self, arr):
        self.weights = arr
        self.outputs = np.array([])
        self.inputs = np.array([])
        self.errors = np.array([])

    def calculate(self, arr, actifunct):
        self.inputs = arr
        res = np.matmul(arr, self.weights)
        if (actifunct == "linear"):
            self.outputs = res
        elif (actifunct == "sigmoid"):
            self.outputs = self.sigmoid(res)
        elif (actifunct == "relu"):
            self.outputs = self.relu(res)
        elif (actifunct == "softmax"):
            self.outputs = self.softmax(res)
        return self.outputs

    def sigmoid(self, arr):
        return 1/(1+np.exp(-arr))
    
    def relu(self, arr):
        return np.maximum(arr,0)

    def softmax(self, arr):
        return np.exp(arr)/np.sum(np.exp(arr))

    def error_softmax(self, num):
        return -np.log(num)
    
    def d_linear(self):
        self.outputs.fill(1)
        return self.outputs

    def d_relu(self):
        self.outputs[self.outputs >= 0] = 1
        self.outputs[self.outputs < 0] = 0
        return self.outputs
    
    def d_sigmoid(self):
        sigm = self.sigmoid(self.outputs)
        return sigm * (1 - sigm)

    def calculate_gradient_softmax(self, target):
        transposed = np.transpose(self.inputs)
        hasil = (self.outputs - target) * transposed
        hasil_transposed = np.transpose(hasil)
        return hasil_transposed
        # return (self.outputs - target) * self.inputs

    def update_weight_output_softmax(self, learning_rate, target):
        
        gradient = self.calculate_gradient_softmax(target)
        print("perhitungan",self.weights[0] - learning_rate * gradient[0])
        for i in range(len(self.errors)):
            self.errors = np.append(self.errors, self.error_softmax(self.outputs[i])) 
        for i in range(len(self.weights)):
            print(len(self.weights)) # 3 karena gradiennya 3*5 jadi mungkin ambil 1 baris lagi?
            print(self.weights[i])
            print(learning_rate)
            print(gradient[i])
            self.weights[i] = self.weights[i] - (learning_rate * gradient[i])
 
    def update_weight_hidden(self, learning_rate, actifunct, error, weight): # error dari layer di depannya
        # calculate errors and gradients
        self.errors = np.array([])
        for i in range(len(error)):
            self.errors = np.append(self.errors, np.multiply(error[i], weight))
        gradient = self.calculate_gradient(actifunct)
        sum_gradient = np.sum(gradient, axis = 0)
        # update weight
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * sum_gradient[i]

    def update_weight_output(self, learning_rate, actifunct, target):
        # calculate errors and gradients
        self.errors = np.subtract(target, self.outputs)
        gradient = self.calculate_gradient(actifunct)
        sum_gradient = np.sum(gradient, axis = 0)
        # update weight
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * sum_gradient[i]

    def calculate_gradient(self, actifunct):
        if (actifunct == "linear"):
            return -self.errors * self.d_linear() * np.transpose(self.inputs)
        elif (actifunct == "relu"):
            return -self.errors * self.d_relu() *  np.transpose(self.inputs)
        elif (actifunct == "sigmoid"):
            return -self.errors * self.d_sigmoid() * np.transpose(self.inputs)
        # elif (actifunct == "softmax"):
        #     return self.d_softmax() * self.inputs

if __name__ == "__main__":
    # perc = Perceptron(np.array([1,2,3]))
    # arr1 = np.array([1,0,0])
    # arr2 = np.array([1,0,1])
    # arr3 = np.array([1,1,0])
    # arr4 = np.array([1,1,1])
    # print(perc.calculate_(np.array([arr1,arr2,arr3,arr4]), "sigmoid"))
    a = [0.5,0.5,0.5,0.5]
    temp = np.empty((0,len(a)), float)
    temp = np.append(temp,([a]),axis=0)
    # temp2 = np.array([0.5],)
    # print(np.append(temp,temp2))
    print(temp)
    # a = np.array([])
    # b = np.append(np.array([1,2,3,4]))
    # print(np.append(a,b))