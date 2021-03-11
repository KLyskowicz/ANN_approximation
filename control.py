from Network import Network 
import os
import numpy as np
import csv

topology = [1,6,1]
activation_fun_topology = ['ident','sigmoid','ident']

training_data = []

fromFile = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '\\' + 'training_data_1.csv', delimiter=' ')
training_data = fromFile[:,0:2]

fromFile = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '\\' + 'test_data.csv', delimiter=' ')
test_data = fromFile[:,0:2]

if not os.path.exists('out'):
    os.makedirs('out')
path = os.path.join(os.getcwd(), 'out')

#          layers_topology, activacion_func_topology, momentum=0.2, learning_rate=0.1, bias=1, epoches=1000, error_measure_frequency=10
net = Network(topology,      activation_fun_topology,      0.2,          0.05,            1,       1000,               1)

#         learning_data, prefiks, path, extend_record=False
net.train(training_data, '1', path, True)

#         learning_data, prefiks, path, extend_record=False
net.test(test_data, '1', path, True)

#              learning_data, path, name
net.print_plot(training_data, path, 'final_plot')