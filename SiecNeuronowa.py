import copy
import math
import random

import numpy as np

from Neuron import Neuron


class SiecNeuronowa:

    def __init__(self, alfa: float, ileWejsc: int, ileNeuronowUkrytych: int, ileNeuronowWyjsciowych: int, bias: bool):
        self.__bias = bias
        self.__alfa = alfa
        self.warstwaUkryta = [Neuron(ileWejsc, bias) for _ in range(ileNeuronowUkrytych)]
        self.warstwaWyjsciowa = [Neuron(ileNeuronowUkrytych, bias) for _ in range(ileNeuronowWyjsciowych)]

    def getNeuronyUkryte(self):
        return self.warstwaUkryta

    def getNeuronyWyjsciowe(self):
        return self.warstwaWyjsciowa

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
        wartosciUkryte = [self.sigmoid(x.sumuj(przypadekUczenia[0])) for x in self.warstwaUkryta]
        # print(wartosciUkryte)

        # wartosci obliczen neuronow wyjsciowych 1 neuron, 1 wyjscie
        wartosciWyjsciowe = [self.sigmoid(x.sumuj(wartosciUkryte)) for x in self.warstwaWyjsciowa]
        # print(wartosciWyjsciowe)

        #####################################
        #       PROPAGACJA WSTECZNA         #
        #####################################
        # LICZENIE BLEDOW, AKTUALIZACJA WAG #
        #####################################

        # bledy
        bledyWyjsciowe = self.obliczajBledyOstatniejWarstwy(wartosciWyjsciowe, przypadekUczenia[1])
        # print("bledy na wyjsciu: ", [round(x,3) for x in bledyWyjsciowe])
        koszt_wyjscia = self.funkcjaKosztu(bledyWyjsciowe)
        # print("koszt wyjscia: ", koszt_wyjscia)

        bledyUkryte = self.obliczajBledyUkrytejWarstwy(bledyWyjsciowe)
        # print("bledy ukryte: ", [round(x,4) for x in bledyUkryte])
        koszt_ukryty = self.funkcjaKosztu(bledyUkryte)
        # print("koszt ukryty: ", koszt_ukryty)


        # aktualizacja wag neuronow ukrytych
        for i in range(len(self.warstwaUkryta)):
            for j in range(len(przypadekUczenia[0])):
                waga = self.warstwaUkryta[i].getWaga(j)
                x = przypadekUczenia[0][j]
                d = bledyUkryte[i]
                a = wartosciUkryte[i]
                # dc/dd = 2d
                pochodna = 2 * d
                # dd/dy = 1
                # dy/da = sigmoid(a) * (1 - sigmoid(a))
                pochodna = (self.sigmoid(a) * (1 - self.sigmoid(a))) * pochodna

                # jezeli uzyto bias tez zaktualizuj go
                if(self.__bias):
                    # da/db = 1
                    bias = self.warstwaUkryta[i].getBias()
                    nowy_bias = bias - (self.__alfa * pochodna)
                    self.warstwaUkryta[i].setBiasWaga(nowy_bias)
                # da/dw = x
                pochodna = pochodna * x
                # pochodna = dc/dw
                nowa_waga = waga - (self.__alfa * pochodna)
                self.warstwaUkryta[i].setWaga(j, nowa_waga)


        # aktualizacja wag neuronow wyjsciowych
        for i in range(len(self.warstwaWyjsciowa)):
            for j in range(len(self.warstwaUkryta)):
                waga = self.warstwaWyjsciowa[i].getWaga(j)
                x = wartosciUkryte[j]
                d = bledyWyjsciowe[i]
                a = wartosciWyjsciowe[i]
                # pochodna dc/dd = 2d
                pochodna = 2 * d
                # pochodna dd/dy = 1
                # pochodna dy/da = sigmoid(a) * (1 - sigmoid(a))
                pochodna = (self.sigmoid(a) * (1 - self.sigmoid(a))) * pochodna

                # jezeli uzyto bias tez zaktualizuj go
                if(self.__bias):
                    # da/db = 1
                    bias = self.warstwaWyjsciowa[i].getBias()
                    nowy_bias = bias - (self.__alfa * pochodna)
                    self.warstwaWyjsciowa[i].setBiasWaga(nowy_bias)

                # pochodna da/dw = x
                pochodna = pochodna * x
                # pochodna = dc/dw
                nowa_waga = waga - (self.__alfa * pochodna)
                self.warstwaWyjsciowa[i].setWaga(j, nowa_waga)



    def obliczajBledyOstatniejWarstwy(self, wyjscieObliczone, wyjscieZnane):
        return [wyjscieObliczone[i] - wyjscieZnane[i] for i in range(len(wyjscieObliczone))]

    def obliczajBledyUkrytejWarstwy(self, bledyWarstwyOstatniej):
        bledyWarstwyUkrytej = [0] * len(self.warstwaUkryta) # tyle bledow ile neuronow ukrytych

        for i in range(len(self.warstwaUkryta)):  # dla kazdego neuronu ukrytego
            for j in range(len(bledyWarstwyOstatniej)):  # zrob cos z bledem warstwy ostatniej
                bledyWarstwyUkrytej[i] += bledyWarstwyOstatniej[j] * self.warstwaWyjsciowa[j].getWaga(i)

        return bledyWarstwyUkrytej

    def test(self, wejscie):
        # wartosci obliczen neuronow ukrytych 1 neuron, 1 wyjscie
        wartosciUkryte = [self.sigmoid(x.sumuj(wejscie)) for x in self.warstwaUkryta]
        # print(wartosciUkryte)
        # wartosci obliczen neuronow wyjsciowych 1 neuron, 1 wyjscie
        wartosciWyjsciowe = [self.sigmoid(x.sumuj(wartosciUkryte)) for x in self.warstwaWyjsciowa]
        return wartosciWyjsciowe

    # Funkcja aktywacyjna - funkcja Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def funkcjaKosztu(self, bledy):
        suma = 0
        for x in bledy:
            suma += x ** 2
        return suma
