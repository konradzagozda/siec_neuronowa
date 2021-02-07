from SiecNeuronowa import SiecNeuronowa


class SiecNeuronowaKlasyfikator(SiecNeuronowa):
    def test(self, wejscie):
        # wartosci obliczen neuronow ukrytych 1 neuron, 1 wyjscie
        wartosciUkryte = [self.sigmoid(x.sumuj(wejscie)) for x in self.warstwaUkryta]
        # print(wartosciUkryte)
        # wartosci obliczen neuronow wyjsciowych 1 neuron, 1 wyjscie
        wartosciWyjsciowe = [0 if self.sigmoid(x.sumuj(wartosciUkryte)) < 0.51 else 1 for x in self.warstwaWyjsciowa]
        return wartosciWyjsciowe