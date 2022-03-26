from layer import Layer
from perceptron import Perceptron
import numpy as np
from sklearn.datasets import load_iris

class Model:
    def __init__(self, filename):
        self._readModel(filename)
        self.model = self._buildModel()

    def __str__(self):
        strprint = "Berikut merupakan struktur dari model:\n"
        for i,layer in enumerate(self.model):
            strprint +="Layer {}:\n".format(i)
            strprint +="Activation Function: {}\n".format(layer.activation_function_type)
            for j,perceptron in enumerate(layer.array_of_perceptron):
                strprint +="  Perceptron {}:\n".format(j)
                for k,weight in enumerate(perceptron.weights):
                    strprint +="   Weight {}: {}\n".format(k,weight)
        return strprint
 
    def _readModel(self, filename):
        arr_act_func = []
        arr_act_perc = []
        with open(filename,"r") as f:
            batch_size = int(f.readline())
            total_layer = int(f.readline())
            total_input = int(f.readline())
            for _ in range(total_layer - 1):
                layer_arr = []
                n_perceptron = int(f.readline())
                arr_act_func.append(f.readline().rstrip("\n"))
                for _ in range(n_perceptron):
                    arr = [int(x) for x in f.readline().split(" ")]
                    new_perceptron = Perceptron(np.array(arr))
                    layer_arr.append(new_perceptron)
                arr_act_perc.append(layer_arr)
        self.batch_size = batch_size
        self.total_layer = total_layer
        self.total_input = total_input
        self.arr_act_func = arr_act_func
        self.arr_act_perc = arr_act_perc

    def _buildModel(self):
        arr_layer = []
        for i in range(self.total_layer-1):
            if (i == total_layer-2): # layer output?
                    arr_layer.append(Layer("output", self.arr_act_func[i], self.arr_act_perc[i]))
                else: # layer hidden?
                    arr_layer.append(Layer("hidden", self.arr_act_func[i], self.arr_act_perc[i]))
        return arr_layer
    
    def doffnn(self,inputarr):
        arr = inputarr
        for layer in self.model:
            bias = [1]*len(inputarr[0])
            # bias = [1,1,1,1]
            arr = np.append([bias],arr,0)
            arr = layer.calculate_all(np.transpose(arr))
        return arr

if __name__ == "__main__":
    iris_X, iris_y = load_iris(return_X_y = True)
    print(iris_X[10], iris_y[10])
    print(np.append(iris_X[10], iris_y[10]))
    # bagi iris_X (dan iris_y) sesuai batch_size
    batch_size = 5 # self?
    n_batch = int(np.ceil(len(iris_X)/batch_size))
    iris_X_batches = [[0 for j in range (batch_size)] for i in range (n_batch)]
    for i in range (n_batch):
        for j in range (batch_size):
            if (5*i+j < len(iris_X)):
                iris_X_batches[i][j] = iris_X[5*i+j]
            else:
                iris_X_batches[i][j] = 0 # handle batch kekurangan anggota
    for i in range (len(iris_X_batches)):
        print(iris_X_batches[i])
    # apakah yg namanya batch itu gini ._.
    # for i in range(epoch):
    #     no_of_batches = len(X_train) // N
    #     for j in range(no_of_batches):
