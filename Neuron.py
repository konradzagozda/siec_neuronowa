import numpy as np


class Neuron:
    # Neuron jest czescia sieci neuronowej, to on wie o swoich wagach
    # na razie robimy bez biasu


    def __init__(self, ileWejsc: int):
        self.__wagi = np.random.uniform(-0.009, 0.009, ileWejsc);


    def getWagi(self):
        return self.__wagi

    def setWaga(self, index: int, value: float):
        self.__wagi[index] = value

    def sumuj(self, wartosci):
        sum = 0
        for i in range(len(wartosci)):
            sum += self.__wagi[i] * wartosci[i]
        return sum

    def getWaga(self, index):
        return self.__wagi[index]
