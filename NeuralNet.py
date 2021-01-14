from random import random
import tools
import numpy


class Node:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = []
        for x in range(num_inputs):
            weight_c = random() * 2 - 1
            self.weights.append(weight_c)

    def run(self, inputs):
        sum_c = 0
        for input_c, weight_c in zip(inputs, self.weights):
            sum_c += input_c * weight_c
        res = tools.sigmoid(sum_c)
        return res


class Net:
    def __init__(self, num_inputs, hidden, num_outputs):
        # Create Input Layer
        self.input_layer = []
        for x in range(num_inputs):
            node_x = Node(1)
            self.input_layer.append(node_x)

        # Create Hidden Layers
        self.hidden_layers = []
        for c, layer_size_c in enumerate(hidden):
            layer_c = []

            num_inputs_c = 0
            if c == 0:
                num_inputs_c = num_inputs
            else:
                num_inputs_c = len(self.hidden_layers[-1])

            for x in range(layer_size_c):
                node_x = Node(num_inputs_c)
                layer_c.append(node_x)
            self.hidden_layers.append(layer_c)

        # Create Output Layer
        self.output_layer = []
        for x in range(num_outputs):
            num_inputs_hidden_layer = len(self.hidden_layers[-1])
            node_x = Node(num_inputs_hidden_layer)
            self.output_layer.append(node_x)

    def run_layer(self, inputs, layer):
        if layer == self.input_layer:
            outputs = []
            for c, node_c in enumerate(layer):
                output_c = node_c.run([inputs[c]])
                outputs.append(output_c)
        else:
            outputs = []
            for c, node_c in enumerate(layer):
                output_c = node_c.run(inputs)
                outputs.append(output_c)

        return outputs

    def run(self, inputs):
        outputs_c = self.run_layer(inputs, self.input_layer)
        for layer_c in self.hidden_layers:
            outputs_c = self.run_layer(outputs_c, layer_c)
        outputs_c = self.run_layer(outputs_c, self.output_layer)
        return outputs_c
