import copy
import random

import numpy as np

from Neuron import Neuron


class SiecNeuronowa:

    def __init__(self, alfa: float, ileWejsc: int, ileNeuronowUkrytych: int, ileNeuronowWyjsciowych: int):
        self.__alfa = alfa
        self.__warstwaUkryta = [Neuron(ileWejsc) for _ in range(ileNeuronowUkrytych)]
        self.__warstwaWyjsciowa = [Neuron(ileNeuronowUkrytych) for _ in range(ileNeuronowWyjsciowych)]

    def getNeuronyUkryte(self):
        return self.__warstwaUkryta

    def getNeuronyWyjsciowe(self):
        return self.__warstwaWyjsciowa

    def procesUczenia(self, ileRazy, przypadkiUczenia):
        przypadkiUczeniaWorking = copy.deepcopy(przypadkiUczenia)
        for i in range(ileRazy):
            self.epokaUczenia(przypadkiUczeniaWorking)

    def epokaUczenia(self, przypadkiUczenia):
        # epoka zawiera wszystkie iteracje
        random.shuffle(przypadkiUczenia)
        for x in przypadkiUczenia:
            self.iteracjaUczenia(x)


    def iteracjaUczenia(self, przypadekUczenia):
        # 1 iteracja uczy wszystkie neurony jednego przypadku uczenia jednokrotnie
        # przypadek uczenia para dwóch tablic: [X, Y]
        # gdzie X - tablica 4 cech wejsciowych, Y - znana tablica 4 cech wyjściowych
        # dla zadania 1 np. ([1,0,0,0],[1,0,0,0])
        # print(przypadekUczenia)

        ####################################
        #         PROPAGACJA W PRZÓD       #
        ####################################
        #  LICZENIE WARTOSCI WYJSCIOWYCH   #
        ####################################

        # wartosci obliczen neuronow ukrytych 1 neuron, 1 wyjscie
        wartosciUkryte = [self.sigmoid(x.sumuj(przypadekUczenia[0])) for x in self.__warstwaUkryta]
        # print(wartosciUkryte)

        # wartosci obliczen neuronow wyjsciowych 1 neuron, 1 wyjscie
        wartosciWyjsciowe = [self.sigmoid(x.sumuj(wartosciUkryte)) for x in self.__warstwaWyjsciowa]
        # print(wartosciWyjsciowe)

        #####################################
        #       PROPAGACJA WSTECZNA         #
        #####################################
        # LICZENIE BLEDOW, AKTUALIZACJA WAG #
        #####################################

        # bledy
        bledyWyjsciowe = self.obliczajBledyOstatniejWarstwy(wartosciWyjsciowe, przypadekUczenia[1])
        # print(bledyWyjsciowe)

        sumaBledow = self.funkcjaKosztu(bledyWyjsciowe)
        print(sumaBledow)

        bledyUkryte = self.obliczajBledyUkrytejWarstwy(bledyWyjsciowe)
        # print(bledyUkryte)


        # aktualizacja wag neuronow ukrytych
        for i in range(len(self.__warstwaUkryta)):
            for j in range(len(przypadekUczenia[0])):
                waga = self.__warstwaUkryta[i].getWaga(j)
                nowa_waga = waga - (self.__alfa * (self.sigmoid_derivative(przypadekUczenia[0][j]) * bledyUkryte[i]))
                self.__warstwaUkryta[i].setWaga(j, nowa_waga)


        # aktualizacja wag neuronow wyjsciowych
        for i in range(len(self.__warstwaWyjsciowa)):
            for j in range(len(self.__warstwaUkryta)):
                waga = self.__warstwaWyjsciowa[i].getWaga(j)
                nowa_waga = waga - (self.__alfa * (self.sigmoid_derivative(wartosciUkryte[j]) * bledyWyjsciowe[i]))
                self.__warstwaWyjsciowa[i].setWaga(j, nowa_waga)



    def obliczajBledyOstatniejWarstwy(self, wyjscieObliczone, wyjscieZnane):
        return [wyjscieZnane[i] - wyjscieObliczone[i] for i in range(len(wyjscieObliczone))]

    def obliczajBledyUkrytejWarstwy(self, bledyWarstwyOstatniej):
        bledyWarstwyUkrytej = [0] * len(self.__warstwaUkryta)
        for i in range(len(bledyWarstwyOstatniej)):
            for j in range(len(self.__warstwaUkryta)):
                bledyWarstwyUkrytej[j] += self.__warstwaWyjsciowa[i].getWaga(j) * bledyWarstwyOstatniej[i]
        return bledyWarstwyUkrytej

    def test(self, wejscie):
        # liczymy X -> Z
        pass

    # Funkcja aktywacyjna - funkcja Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Pochodna Sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def funkcjaKosztu(self, bledy):
        suma = 0
        for x in bledy:
            suma += x ** 2
        return suma
