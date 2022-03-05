import numpy as np

class Perceptron:

    def __init__(self, arr):
        self.weights = arr

    def calculate(self, arr, actifunct):
        res = np.matmul(arr, self.weights)
        if (actifunct == "linear"):
            return res
        elif (actifunct == "sigmoid"):
            return self.sigmoid(res)
        elif (actifunct == "relu"):
            return self.relu(res)
        elif (actifunct == "softmax"):
            return self.softmax(res)

    def sigmoid(self, arr):
        return 1/(1+np.exp(-arr))
    
    def relu(self, arr):
        return np.maximum(arr,0)

    def softmax(self, arr):
        return np.exp(arr)/np.sum(np.exp(arr))

if __name__ == "__main__":
    perc = Perceptron(np.array([1,2,3]))
    arr1 = np.array([1,0,0])
    arr2 = np.array([1,0,1])
    arr3 = np.array([1,1,0])
    arr4 = np.array([1,1,1])
    print(perc.calculate(np.array([arr1,arr2,arr3,arr4]), "sigmoid"))