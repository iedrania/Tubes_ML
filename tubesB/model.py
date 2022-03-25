from layer import Layer
from perceptron import Perceptron
import numpy as np

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
        self.total_layer = total_layer
        self.total_input = total_input 
        self.arr_act_func = arr_act_func
        self.arr_act_perc = arr_act_perc

    def _buildModel(self):
        arr_layer = []
        for i in range(self.total_layer-1):
            arr_layer.append(Layer(self.arr_act_func[i], self.arr_act_perc[i]))
        return arr_layer
    
    def doffnn(self,inputarr):
        arr = inputarr
        for layer in self.model:
            bias = [1]*len(inputarr[0])
            # bias = [1,1,1,1]
            arr = np.append([bias],arr,0)
            arr = layer.calculate_all(np.transpose(arr))
        return arr