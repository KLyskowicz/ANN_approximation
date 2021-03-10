import random
import os
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron

class Network(object):

    def __init__(self, layers_topology, activacion_func_topology, momentum=0.2, learning_rate=0.1, bias=1, epoches=1000, error_measure_frequency=10):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.bias = bias
        self.epoches = epoches
        self.error_measure_frequency = error_measure_frequency
        self.error = 0
        self.error_X = []
        self.error_Y = []
        self.input_amount = layers_topology[0]
        self.layers = [[Neuron(layers_topology[i-1], neuron_number+1, activacion_func_topology[i], self.momentum, self.learning_rate, self.bias) for neuron_number in range(layers_topology[i])] for i in range(len(layers_topology))]
    
    def predict(self, input):
        for neuron, x in zip(self.layers[0], input):
            neuron.output_value = x
        for layer, previous_layer in zip(self.layers[1:], self.layers[0:]):
            for neuron in layer:
                neuron.predict(previous_layer)

    def error_calculation(self, expected_output):
        for neuron, output in zip(self.layers[-1], expected_output):
            delta = output - neuron.output_value
            self.error += (delta*delta)/2

    def back_propagation(self, expected_output):
        self.error_calculation(expected_output)
        for neuron, output in zip(self.layers[-1], expected_output):
            neuron.output_layer_factor(output)
        for layer, next_layer in zip(reversed(self.layers[1:-1]), reversed(self.layers[2:])):
            for neuron in layer:
                neuron.hidden_layer_factor(next_layer)
        for layer, previous_layer in zip(reversed(self.layers[1:]), reversed(self.layers[0:-1])):
            for neuron in layer:
                neuron.update_weights(previous_layer)

    def get_predicted(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.output_value)
        return output

    def train(self, learning_data, prefiks, path, extend_record=False):
        if(extend_record):
            name = os.path.join(path, str(prefiks) + "_train_error_tracking.txt")
            error_save = open(name,"w+")
            name = os.path.join(path, str(prefiks) + "_train_date_predicting_control.txt")
            data_control = open(name,"w+")
        for i in range(self.epoches):
            self.error = 0
            np.random.shuffle(learning_data)
            X = []
            Y = []
            for row in learning_data:
                X.append(row[0:self.input_amount])
                Y.append(row[self.input_amount:])
            for data_in, data_out in zip(X, Y):
                self.predict(data_in)
                self.back_propagation(data_out)
                if(extend_record):
                    if i % self.error_measure_frequency == 0 or i == 1 or i == self.epoches - 1:
                        data_control.write( "Epoch: " + str(i) + "\n" )
                        data_control.write( " Training input:  " + str(data_in) + "\n")
                        data_control.write( " Expected output: " + str(data_out) + "\n")
                        data_control.write( " Received output: ")
                        for neuron in self.layers[-1]:
                            data_control.write( str(neuron.output_value) + " " )
                        data_control.write("\n")
                        data_control.write( " Approximated output: ")
                        for neuron in self.layers[-1]:
                            data_control.write( str(round(neuron.output_value,3)) + " " )
                        data_control.write("\n")
            if i % self.error_measure_frequency == 0 or i == 1 or i == self.epoches - 1:
                if(extend_record):
                    error_save.write( str(self.error/len(self.layers[-1])) + '\n')
                self.error_Y.append(self.error/len(self.layers[-1]))
                self.error_X.append(i)
        if(extend_record):
            error_save.close()
            data_control.close()
            name = str(prefiks) + '_train_error_plot.png'
            self.print_error_plot(path, name)

    def print_error_plot(self, path, name):
        plt.plot(self.error_X, self.error_Y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Error plot')
        plt.savefig(os.path.join(path, name))
        plt.close()

    def test(self, learning_data, prefiks, path, extend_record=False):
        if(extend_record):
            name = os.path.join(path, str(prefiks) + "_test_error_tracking.txt")
            error_save = open(name,"w+")
            name = os.path.join(path, str(prefiks) + "_test_data_predicted.txt")
            test_data_predicted = open(name,"w+")
            name = os.path.join(path, str(prefiks) + "_test_error_resoult.txt")
            test_error_resoult = open(name,"w+")
        X = []
        Y = []
        for row in learning_data:
            X.append(row[0:self.input_amount])
            Y.append(row[self.input_amount:])
        error_sum = 0
        error_max = 0
        self.error_X.clear()
        self.error_Y.clear()
        for data_in, data_out, i in zip(X, Y, range(len(X))):
            self.error = 0
            self.predict(data_in)
            self.error_calculation(data_out)
            error_sum += self.error
            if error_max < self.error:
                error_max = self.error
            if(extend_record):
                test_data_predicted.write( " Tested input   :     " + str(data_in) + "\n")
                test_data_predicted.write( " Expected output:     " + str(data_out) + "\n")
                test_data_predicted.write( " Received output:     ")
                for neuron in self.layers[-1]:
                    test_data_predicted.write( str(neuron.output_value) + " " )
                test_data_predicted.write("\n")
                test_data_predicted.write( " Approximated output: ")
                for neuron in self.layers[-1]:
                     test_data_predicted.write( str(round(neuron.output_value,6)) + " " )
                test_data_predicted.write("\n")
                error_save.write( str(self.error/len(self.layers[-1])) + '\n')
            self.error_Y.append(self.error/len(self.layers[-1]))
            self.error_X.append(i+1)
        if(extend_record):
            error_sum = error_sum/len(self.layers[-1])
            test_error_resoult.write( " Max error: " + str('%f' % (round(error_max/len(self.layers[-1]),6))) + '\n')
            error_sum = error_sum/len(X)
            test_error_resoult.write( " Avarage error: " + str(round(error_sum,6)))
            test_error_resoult.close()
            error_save.close()
            test_data_predicted.close()
        name = str(prefiks) + '_test_error_plot.png'
        self.print_error_plot(path, name)

    def predict2(self, input):
        for neuron in self.layers[0]:
            neuron.output_value = input
        for layer, previous_layer in zip(self.layers[1:], self.layers[0:]):
            for neuron in layer:
                neuron.predict(previous_layer)

    def print_plot(self, learning_data, path, name):
        x = []
        y = []
        a = 0.01
        b = -6
        for i in range(1200):
            x.append(b)
            b += a
        for i in x:
            self.predict2(i)
            y.append(self.layers[-1][0].output_value)
        plt.plot(x, y, 'g')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Training data and approximated output')
        X2 = []
        Y2 = []
        for row in learning_data:
            X2.append(row[0:1])
            Y2.append(row[1:])
        plt.scatter(X2,Y2)
        plt.savefig(os.path.join(path, str(name)))
