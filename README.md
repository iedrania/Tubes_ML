# TubesA_AI

"linear"
"sigmoid"
"relu"
"softmax"

input:
0 0
0 1
1 0
1 1

model:
3
2
2
20 20
-20 -20
1
20 20

perceptron:
attr:
- array of weight (20, 20)
func:
- calculate(array input, actifunc): ngitung sigma terus pake actifunc sesuai
- actifunc(4)

layer:
attr:
- activation function type
- array of perceptron
function:
- calculate_all(input array): ngitung input array ke array of perceptron, terus masukin ke output array --> for all perceptron: array output [i] = calculate(input array, actifunc)

program:
3 -> jumlah layer
2 -> number of input data
2 -> jumlah neuron di layer 1
----neuron 1
----neuron 2
1 -> jumlah neuron di layer 2
----neuron 1


pilih tipe input: single ato batch
single: 2
batch:
jumlah input: 4
4 * 2

for all layer: calculate_all(input array) -> output array -> calculate_all(output array)