from SiecNeuronowa import SiecNeuronowa


class SiecNeuronowaKlasyfikator(SiecNeuronowa):
    def obliczajBledyOstatniejWarstwy(self, wyjscieObliczone, wyjscieZnane):
        return [wyjscieZnane[i] - wyjscieObliczone[i] for i in range(len(wyjscieObliczone))]