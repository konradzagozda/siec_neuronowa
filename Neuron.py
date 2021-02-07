import numpy as np


class Neuron:
    # Neuron jest czescia sieci neuronowej, to on wie o swoich wagach
    # na razie robimy bez biasu

    def __init__(self, ileWejsc: int, bias: bool):
        self.__bias = bias
        if (bias):
            self.__bias_waga = np.random.uniform(-0.09, 0.09)
        self.__wagi = np.random.uniform(-0.09, 0.09, ileWejsc)

    def getWagi(self):
        return self.__wagi

    def getBias(self):
        return self.__bias_waga

    def setBiasWaga(self, biaswaga):
        self.__bias_waga = biaswaga

    def setWaga(self, index: int, value: float):
        self.__wagi[index] = value

    def sumuj(self, wartosci):
        sum = 0
        for i in range(len(wartosci)):
            sum += self.__wagi[i] * wartosci[i]
        if (self.__bias):
            sum += 1 * self.__bias_waga
        return sum

    def getWaga(self, index):
        return self.__wagi[index]
