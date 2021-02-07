import numpy as np

from SiecNeuronowa import SiecNeuronowa
from SiecNeuronowaKlasyfikator import SiecNeuronowaKlasyfikator
# zadanie 1


przypadki_uczenia = [([1,0,0,0], [1,0,0,0]),
                     ([0,1,0,0], [0,1,0,0]),
                     ([0,0,1,0], [0,0,1,0]),
                     ([0,0,0,1], [0,0,0,1])]

# 1 neuron
siec1neuron = SiecNeuronowa(0.1,4,1,4)
siec1neuron.procesUczenia(10000, przypadki_uczenia)

print("\nsiec z jednym neuronem ukrytym:")
for x in przypadki_uczenia:
    print(np.round(siec1neuron.test(x[0]), 2))

#
# # 2 neurony
# siec2neurony = SiecNeuronowa(0.1,4,2,4)
# siec2neurony.procesUczenia(10000, przypadki_uczenia)
#
# print("\nsiec z dwoma neuromani ukrytymi:")
# for x in przypadki_uczenia:
#     print(np.round(siec2neurony.test(x[0]), 2))
#
# # 3 neurony
#
# siec3neurony = SiecNeuronowa(0.1,4,3,4)
# siec3neurony.procesUczenia(10000, przypadki_uczenia)
#
# print("\nsiec z trzema neuromani ukrytymi:")
# for x in przypadki_uczenia:
#     print(np.round(siec3neurony.test(x[0]), 2))


# # #zadanie 2 xor
# # from SiecNeuronowaKlasyfikator import SiecNeuronowaKlasyfikator
# #
# przypadki_uczenia_xor = [([0,0], [0]),
#                         ([1,0], [1]),
#                         ([0,1], [1]),
#                         ([1,1], [0]),]
# #
# # # 2 neurony
# # siecXOR = SiecNeuronowaKlasyfikator(0.05,2,2,1)
# # siecXOR.procesUczenia(200000, przypadki_uczenia_xor)
# # print("siec (XOR) z dwoma neuronami ukrytymi:")
# # for x in przypadki_uczenia_xor:
# #     print(np.round(siecXOR.test(x[0])))
# #
# #
# # 3 neurony
# siecXOR = SiecNeuronowaKlasyfikator(0.05,2,3,1)
# siecXOR.procesUczenia(10000, przypadki_uczenia_xor)
# print("siec (XOR) z trzema neuronami ukrytymi:")
# for x in przypadki_uczenia_xor:
#     print((siecXOR.test(x[0])))




