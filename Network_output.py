import numpy as np
import random
import pickle
import shutil
import gzip
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class network_output(object):

    def __init__(self, pesos, biases, sizes):
        # Passa parâmetros de uma rede neural
        self.weights = pesos
        self.biases = biases
        self.sizes = sizes
        self.acertos = 0

#------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def feed_forward(self, activations):

        i = 0
        vet_activations = []
        vet_activations.append(activations)

        for x in self.sizes[1:]:
            z = (np.dot(self.weights[i], activations)) + self.biases[i]
            activations = self.sigmoid(z)
            vet_activations.append(activations)
            i += 1

        vet_activations = np.asarray(vet_activations)

        return (vet_activations)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def sigmoid(self, z):
        #retorna a função sigmoid (ativação do neurônio) tendo como entrada o z
        return (1.0 / (1.0 + np.exp(-z)))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def rede_neural(self):
        #Cria um zip com vetor chamado rede_neural, que armazena as matrizes de pesos e os vetores de biases e sizes da network

        rede_neural = self.weights,self.biases,self.sizes
        pickle_out = open("rede_neural.pkl", "wb")
        pickle.dump(rede_neural, pickle_out)
        pickle_out.close()

        with open('rede_neural.pkl', 'rb') as f_input:
            with gzip.open('rede_neural.pkl.gz', 'wb') as f_output:
                shutil.copyfileobj(f_input, f_output)

        os.remove('rede_neural.pkl')
        return
# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def testes_aleat(self,vetor_dados,testes):
        # Função que faz testes aleatórios de imagens e verifica se o output previu corretamente
        a = 0
        b = len(vetor_dados[1])
        for x in range(testes):
            numero = random.randint(0, b)
            activations = self.feed_forward(vetor_dados[0][numero].reshape(-1, 1))
            # print(activations[-1]) ## Exibe o output da rede neural de correspondente ao input de uma imagem
            #print("esse número aí me parece um {}".format(np.argmax(activations[-1]))) #Qual número a rede neural identificou
            #print("A resposta certa é: {}".format(np.argmax(vetor_dados[1][numero]))) #Qual era a resposta correta
            if (np.argmax(activations[-1]) != np.argmax(vetor_dados[1][numero])):
                ##Plot da imagem que a rede errou
                #plt.imshow(vetor_dados[0][numero].reshape((28, 28)), cmap=cm.Greys_r)
                #plt.show()
                a += 1

        print((100 - (a / testes) * 100), "% de acerto")