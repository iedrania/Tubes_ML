from layer import Layer
from perceptron import Perceptron
import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

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
                    arr = [float(x) for x in f.readline().split(" ")]
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
            if (i == self.total_layer-2): # layer output
                arr_layer.append(Layer("output", self.arr_act_func[i], self.arr_act_perc[i]))
            else: # layer hidden
                arr_layer.append(Layer("hidden", self.arr_act_func[i], self.arr_act_perc[i]))
        return arr_layer
    
    def saveModel(self,filename):
        res = ""
        nl = "\n"
        res += str(self.batch_size) + nl
        res += str(self.total_layer) + nl
        res += str(self.total_input) + nl
        for layer in self.model:
            res += str(len(layer.array_of_perceptron)) + nl
            res += (layer.activation_function_type) + nl
            for i in range(len(layer.array_of_perceptron)):
                p = layer.array_of_perceptron[i]
                res += " ".join([str(x) for x in (p.weights).tolist()]) + nl
        with open(filename, "w") as f:
            f.write(res[:-1])

    def doffnn(self,inputarr):
        arr = np.transpose(inputarr)
        arr = arr.astype('longdouble')
        for layer in self.model:
            bias = [1]*len(arr[0])
            arr = np.append([bias],arr,0)
            arr = layer.calculate_all(np.transpose(arr))
        return arr
    
    def dobackwardpropagation(self, learning_rate, target):
        for i in reversed(range(len(self.model))) :
            layer = self.model[i]
            if (layer.type == "output"):
                temp_error, weight_before = layer.calculate_back_output(learning_rate, target)
            else:
                temp_error = layer.calculate_back_hidden(learning_rate, temp_error, weight_before)

    def fit(self,inputarr, target, learningrate,errorthreshold, maxiter):
        loss = 9999999
        for j in range(maxiter):
            outputs = []
            for i in range (len(inputarr)):
                output = self.doffnn(inputarr[i])
                outputs.append(output[0])
                self.dobackwardpropagation(learningrate, target[i])
            loss = self.crossentropy(output) if (self.model[-1].activation_function_type=="softmax") else self.sumsquarederror(output,target)
            # # print("ini adalah output ke", j + 1, "=", output)
            # print("ini adalah loss ke", j + 1, "=", loss)
        return outputs
            

    def sumsquarederror(self,outputs,target): # target = iris_y
        sigma = 0
        for i in range(len(outputs)):
            for j in range(len(outputs)):
                sigma += target[i][j] - outputs[i][j]
            break
        return 0.5*sigma**2

    # def crossentropy(self,output):
    #     return -np.log(output)

    def printw(self):
         for layer in self.model:
             for p in layer.array_of_perceptron:
                 print(p.weights)

if __name__ == "__main__":
    
    iris_X, iris_y = load_iris(return_X_y = True)
    # iris_X, iris_y = shuffle(iris_X, iris_y, random_state=0)
    # print(iris_X[10], iris_y[10])
    # print(np.append(iris_X, iris_y))
    # # bagi iris_X (dan iris_y) sesuai batch_size
    # batch_size = 5 # self?
    # n_batch = int(np.ceil(len(iris_X)/batch_size))
    # iris_X_batches = [[0 for j in range (batch_size)] for i in range (n_batch)]
    # for i in range (n_batch):
    #     for j in range (batch_size):
    #         if (batch_size*i+j < len(iris_X)):
    #             np.append(iris_X_batches[i], iris_X[5*i+j])
    #         else:
    #             iris_X_batches[i][j] = 0 # todo handle batch kekurangan anggota
    
    # bikin mini batch
    model = Model("model.txt")
    model.saveModel("test.txt")
    # mini_batch_X = []
    # mini_batch_y = []
    # batches_X = []
    # batches_y = []
    
    # for i in range (len(iris_X)):
    #     mini_batch_X.append(iris_X[i])
    #     mini_batch_y.append(iris_y[i])
    #     if (i % model.batch_size == model.batch_size - 1): # isi mini_batch sebanyak batch_size
    #         batches_X.append(mini_batch_X)
    #         batches_y.append(mini_batch_y)
    #         mini_batch_X = []
    #         mini_batch_y = []
    # model.fit(batches_X, batches_y, 0.1, 1, 20)

    # for i in range (len(batches_X)):
    #     print(i)
    #     print(batches_X[i])
    #     print(batches_y[i])
    # apakah yg namanya batch itu gini ._.
    # for i in range(epoch):
    #     no_of_batches = len(X_train) // N
    #     for j in range(no_of_batches):
    
    # arr1 = np.array([0,0,1,1])
    # arr2 = np.array([0,1,0,1])
    # array_input = np.array([arr1, arr2])
    # target = np.array([1,0,0,1])
    # model = Model("model.txt")
    # model.fit(array_input, target, 0.05,0.01,10)
