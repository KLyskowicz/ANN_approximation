import random
import numpy as np

class Neuron(object):                                             #sigmoid; ident
    def __init__(self, inputs_amount, neuron_number_in_layer, activation_function_choice, momentum=0.2, learning_rate=0.1, bias=1):
        self.bias = bias
        self.weights = np.zeros(inputs_amount + 1)
        for i in range(inputs_amount + 1):
            self.weights[i] = random.uniform(-0.5,0.5)
        if self.bias:
            self.weights[0] = 1
        self.weight_delta = np.zeros(inputs_amount + 1)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.output_value = 0.0
        self.weight_change_factor = 0.0
        self.number = neuron_number_in_layer
        self.activation_function_chice = activation_function_choice

    def activation_function_sigmoid(self, stimulus):
        return 1 / (1 + np.exp(-stimulus))
        
    def activation_function_ident(self, stimulus):
        return stimulus

    def activation_function(self, stimulus):
        if self.activation_function_chice == 'sigmoid':
            return self.activation_function_sigmoid(stimulus)
        elif self.activation_function_chice == 'ident':
            return self.activation_function_ident(stimulus)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def identical_derivative(self, x):
        return 1

    def activation_function_derivative(self, x):
        if self.activation_function_chice == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_function_chice == 'ident':
            return self.identical_derivative(x)        

    def set_output_value(self, value):
        self.output_value = value

    def predict(self, previous_layer):
        summation = 0
        for weight, neuron in zip(self.weights[1:], previous_layer):
            x = weight * neuron.output_value
            summation += x
        if self.bias:
            summation += self.weights[0]
        self.output_value = self.activation_function(summation)

    def output_layer_factor(self, expected_value):
        self.weight_change_factor = (self.output_value - expected_value) * self.activation_function_derivative(self.output_value)

    def hidden_layer_factor(self, next_layer):
        summation = 0
        for neuron in next_layer:
            summation += neuron.weight_change_factor * neuron.weights[self.number]
        self.weight_change_factor = summation * self.activation_function_derivative(self.output_value)

    def update_weights(self, previous_layes):
        for previous_neuron, i in zip(previous_layes, range(1, len(self.weights))):
            delta_weight = self.learning_rate * previous_neuron.output_value * self.weight_change_factor + self.momentum * self.weight_delta[i]
            self.weight_delta[i] = delta_weight
            self.weights[i] -= delta_weight
        if self.bias:
            delta_weight = self.learning_rate * 1 * self.weight_change_factor + self.momentum * self.weight_delta[0]
            self.weight_delta[0] = delta_weight
            self.weights[0] -= delta_weight