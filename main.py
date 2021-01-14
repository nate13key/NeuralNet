import NeuralNet
from random import random


def test_1():
    x = NeuralNet.Node(1)
    inputs = []
    for c in range(x.num_inputs):
        input_val_c = random()
        inputs.append(input_val_c)
    inputs = [0]
    res = x.run(inputs)
    print(res)


def main():
    greater_than_bot = NeuralNet.Net(2, [3, 3], 1)
    ans = greater_than_bot.run([2, 3])
    print(ans)


if __name__ == '__main__':
    main()
