import numpy as np

from SiecNeuronowa import SiecNeuronowa

# zadanie 1


przypadki_uczenia = [([1,0,0,0], [1,0,0,0]),
                     ([0,1,0,0], [0,1,0,0]),
                     ([0,0,1,0], [0,0,1,0]),
                     ([0,0,0,1], [0,0,0,1])]

# 1 neuron bez bias
siec1neuron = SiecNeuronowa(0.1,4,1,4, False)
siec1neuron.procesUczenia(50000, przypadki_uczenia)

print("\nsiec z jednym neuronem ukrytym bez bias:")
for x in przypadki_uczenia:
    print(np.round(siec1neuron.test(x[0]), 2))

# 1 neuron z bias
siec1neuron = SiecNeuronowa(0.1,4,1,4, True)
siec1neuron.procesUczenia(50000, przypadki_uczenia)

print("\nsiec z jednym neuronem ukrytym z bias:")
for x in przypadki_uczenia:
    print(np.round(siec1neuron.test(x[0]), 2))


# 2 neurony bez bias
siec2neurony = SiecNeuronowa(0.1,4,2,4, False)
siec2neurony.procesUczenia(50000, przypadki_uczenia)

print("\nsiec z dwoma neuromani ukrytymi bez bias:")
for x in przypadki_uczenia:
    print(np.round(siec2neurony.test(x[0]), 2))

# 2 neurony z bias
siec2neurony = SiecNeuronowa(0.1,4,2,4, True)
siec2neurony.procesUczenia(50000, przypadki_uczenia)

print("\nsiec z dwoma neuromani ukrytymi z bias:")
for x in przypadki_uczenia:
    print(np.round(siec2neurony.test(x[0]), 2))

# 3 neurony bez bias

siec3neurony = SiecNeuronowa(0.1,4,3,4, False)
siec3neurony.procesUczenia(50000, przypadki_uczenia)

print("\nsiec z trzema neuromani ukrytymi bez bias:")
for x in przypadki_uczenia:
    print(np.round(siec3neurony.test(x[0]), 2))

# 3 neurony z bias

siec3neurony = SiecNeuronowa(0.1,4,3,4, True)
siec3neurony.procesUczenia(50000, przypadki_uczenia)

print("\nsiec z trzema neuromani ukrytymi z bias:")
for x in przypadki_uczenia:
    print(np.round(siec3neurony.test(x[0]), 2))


# #zadanie 2 xor
from SiecNeuronowaKlasyfikator import SiecNeuronowaKlasyfikator

przypadki_uczenia_xor = [([0,0], [0]),
                        ([1,0], [1]),
                        ([0,1], [1]),
                        ([1,1], [0])]

# 2 neurony
siecXOR = SiecNeuronowaKlasyfikator(0.1,2,2,1, False)
siecXOR.procesUczenia(50000, przypadki_uczenia_xor)
print("\nsiec (XOR) z dwoma neuronami ukrytymi:")
for x in przypadki_uczenia_xor:
    print(siecXOR.test(x[0]))


# 3 neurony
siecXOR = SiecNeuronowaKlasyfikator(0.1,2,3,1, False)
siecXOR.procesUczenia(50000, przypadki_uczenia_xor)
print("\nsiec (XOR) z trzema neuronami ukrytymi:")
for x in przypadki_uczenia_xor:
    print((siecXOR.test(x[0])))




