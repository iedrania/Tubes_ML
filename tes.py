from model import Model
import numpy as np
model = Model("model.txt")

def bacainput():
    arr1 = np.array([0,0,1,1])
    arr2 = np.array([0,1,0,1])
    return np.array([arr1, arr2])


arr = [[0],[1]]
print(model.doffnn(arr))