# inputs = [1, 2, 3, 2.5]

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]

# bias = [2, 3, 0.5]

#---------------------------------------- Static way of doing it --------------------------------------------------#
'''
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

print(output)
'''
#------------------------------------------------------------------------------------------------------------------#


# ------------------------------------- Dynamic way of doing it ---------------------------------------------------#
'''
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

layer_outputs = []  # Output of current layer
for neuron_weights, neuron_bias in zip(weights, bias): # zip is used to iterate over multiple lists at once 
    neuron_output = 0  # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight # Multiply this input by associated weight and add to the neuron's output variable
    neuron_output += neuron_bias # Add bias to the neuron's output
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''
#------------------------------------------------------------------------------------------------------------------#

# ------------------------------------- Using Numpy for only one neuron -------------------------------------------#
'''
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(weights, inputs) + bias # np.dot() is used to multiply two arrays
print(output)
'''
#------------------------------------------------------------------------------------------------------------------#

# ------------------------------------- Using Numpy for a layer of neurons ----------------------------------------#
'''
import numpy as np
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)
'''
#------------------------------------------------------------------------------------------------------------------#

# ------------------------------------- Using Numpy for a batch of data -------------------------------------------#
'''
import numpy as np
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

print(np.array(inputs).shape)
print(np.array(weights).shape)
print(np.array(weights).T.shape)

biases = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + biases 
print(output)
'''
#------------------------------------------------------------------------------------------------------------------#

# ------------------------------- Using Numpy for a batch of data (2 layers) --------------------------------------#
'''
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases1 = [2, 3, 0.5]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
'''
#------------------------------------------------------------------------------------------------------------------#

# ------------------------------- Using Numpy for a batch of data (2 layers) --------------------------------------#
'''
import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # 0.10 is used to scale the values to a useful range
        # np.random.randn() is used to create an array of random numbers 
        # n_inputs is the number of inputs and n_neurons is the number of neurons
        # so the shape of the array will be (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # np.zeros() is used to create an array of zeros # shape = (1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# print(0.10*np.random.randn(4, 3))

layer1 = Layer_Dense(4, 5) # 4 inputs, 5 neurons
layer2 = Layer_Dense(5, 2) # 5 inputs, 2 neurons

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
'''
#------------------------------------------------------------------------------------------------------------------#

# ----------------------------------- Usage of Activation Function (ReLU) -----------------------------------------#
'''
import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# ReLU Activation Function
''' '''
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
''' '''
for i in inputs:                # Smatter way of doing it
    output.append(max(0, i))

print(output)
'''
# -----------------------------------------------------------------------------------------------------------------#

# ----------------------------------- Usage of Activation Function (ReLU) -----------------------------------------#
'''
import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # 0.10 is used to scale the values to a useful range
        # np.random.randn() is used to create an array of random numbers 
        # n_inputs is the number of inputs and n_neurons is the number of neurons
        # so the shape of the array will be (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # np.zeros() is used to create an array of zeros # shape = (1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4, 5) # 4 inputs, 5 neurons

activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)
'''
# -----------------------------------------------------------------------------------------------------------------#

# ----------------------------------- Usage of Activation Function(RELU) nnfs--------------------------------------#
'''
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(300, 3)
# print (X,y)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # 0.10 is used to scale the values to a useful range
        # np.random.randn() is used to create an array of random numbers 
        # n_inputs is the number of inputs and n_neurons is the number of neurons
        # so the shape of the array will be (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # np.zeros() is used to create an array of zeros # shape = (1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# print(0.10*np.random.randn(4, 3))

layer1 = Layer_Dense(2, 455) # 4 inputs, 5 neurons
# layer2 = Layer_Dense(455, 2) # 5 inputs, 2 neurons

# print(layer1.biases)

activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)

# layer2.forward(activation1.output)
# print(layer2.output)

# activation1.forward(layer2.output)
# print(activation1.output)
'''
# -----------------------------------------------------------------------------------------------------------------#

# ----------------------------------- Usage of Activation Function(Softmax) ---------------------------------------#

import numpy as np  
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # 0.10 is used to scale the values to a useful range
        # np.random.randn() is used to create an array of random numbers 
        # n_inputs is the number of inputs and n_neurons is the number of neurons
        # so the shape of the array will be (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # np.zeros() is used to create an array of zeros # shape = (1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # np.max() is used to find the maximum value in an array
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # np.sum() is used to find the sum of all the values in an array
        self.output = probabilities

layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])

# print(np.sum(activation2.output, axis=1, keepdims=True))

