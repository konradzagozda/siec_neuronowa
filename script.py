from Neuron import Neuron
from SiecNeuronowa import SiecNeuronowa

przypadki_uczenia = [([1,0,0,0], [1,0,0,0]),
                     ([0,1,0,0], [0,1,0,0]),
                     ([0,0,1,0], [0,0,1,0]),
                     ([0,0,0,1], [0,0,0,1])]


siec = SiecNeuronowa(0.1,4,2,4)

siec.procesUczenia(10000, przypadki_uczenia)



