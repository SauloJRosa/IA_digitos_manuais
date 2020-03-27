import numpy as np
import random
import pickle
import shutil
import gzip
import os

class Network(object):

    def __init__(self,sizes):
        self.num_layers = len(sizes) #comprimento do vetor de parâmetro (sizes) dará o número de camadas da rede (contando com input e output layer)
        self.sizes = sizes #coletando o vetor de parâmetro em um vetor dentro do método

        # Cria um array contendo arrays que armazenam os bias dos neurônios de cada layer
        # Para acessar o bias de um neurônio: self.biases[número da hidden layer][número do neurônio na hidden layer]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # Cria um array 2D contendo os pesos de cada layer para a outra. se tivermos 3 camadas, terá 2 matrizes de pesos, conectando cada layer á seguinte
        # Para acessar um peso específico: self.weights[número da matriz de peso][número da linha][número da coluna]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.acertos = 0

#------------------------------------------------------------------------------------------------------------------------------------------------------------#

    def SGD(self, vetor_dados, epochs, mini_batch_size, eta):

        #Variáveis necessárias para criar os mini_batches
        a = mini_batch_size
        n = len(vetor_dados[1])
        b = 0
        cont = 1
        mini_batches = []
        vet_epochs = []

        #Embaralhando as imagens e seus respectivos labbels
        tmp = list(zip(vetor_dados[0], vetor_dados[1]))
        random.shuffle(tmp)
        vetor_dados[0], vetor_dados[1] = zip(*tmp)

        #Criando os mini_batches. O número de mini_batches será a quantidade de imagens carregadas dividido pelo tamanho
        #de cada mini_batch
        while (cont <= (n/a)):
            mini_batch = vetor_dados[0][b:mini_batch_size], vetor_dados[1][b:mini_batch_size]
            b = mini_batch_size
            mini_batches.append(mini_batch)
            mini_batch_size = mini_batch_size + a
            cont += 1
        mini_batches = np.asarray(mini_batches)

        for x in range(epochs):
            #Treinando com cada mini_batch do vetor mini_batches
            for mini_batch in mini_batches:
                self.atualiza_mini_batch(mini_batch, eta)
            print("Epoch {0}: {1} / {2} ou seja: {3}%".format(x, self.acertos, n,round(((self.acertos/n)*100), 2)))
            self.acertos = 0

#------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def feed_forward(self,activations):
        #alimentando a rede neural com o vetor de inputs e devolvendo como output o vetor de ativações de cada camada
        i = 0
        vet_activations = []
        z_vector = []
        vet_activations.append(activations)

        for x in self.sizes[1:]:
            z = (np.dot(self.weights[i], activations)) + self.biases[i]
            z_vector.append(z)
            activations = self.sigmoid(z)
            vet_activations.append(activations)
            i += 1

        z_vector = np.asarray(z_vector)
        vet_activations = np.asarray(vet_activations)

        return (vet_activations, z_vector)

#------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def back_propagation(self,activations,referencia):
        #criando os vetores/matrizes de derivadas parciais e preenchendo de 0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Faz o FeedForward e pega os 2 vetores gerados, o vet_activations e o z_vector
        vet_activations, z_vector = self.feed_forward(activations)
        delta = (vet_activations[-1] - referencia) * self.sigmoid_derivate(z_vector[-1])

        if (np.argmax(vet_activations[-1]) == np.argmax(referencia)):
            self.acertos += 1

        #Pega o primeiro vetor de derivadas de bias e a primeira matrix de derivadas de pesos (weight)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, vet_activations[-2].transpose())

        #Faz um loop para preencher o vetor e a matriz de derivadas de bias e pesos de cada camada partindo da penúltima
        #para a primeira
        for x in range (2, self.num_layers):
            z = z_vector[-x]
            delta = np.dot(self.weights[-x+1].transpose(),delta)*(self.sigmoid_derivate(z))
            nabla_b[-x] = delta
            nabla_w[-x] = np.dot(delta, vet_activations[-x-1].transpose())

        return (nabla_b, nabla_w)

#------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def atualiza_mini_batch(self, mini_batch, eta):

        mini_batch = np.asarray(mini_batch)

        #criando vetores de bias e matrizes de pesos com 0 para posterior preenchimento
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Somando os vetores e matrizes das derivadas de bias e pesos após o backpropagation de cada imagem de treino que
        #compõe o mini_batch

        x = 0
        for x in range((len(mini_batch[1]-1))):
            delta_nabla_b, delta_nabla_w = self.back_propagation(mini_batch[0][x].reshape(-1, 1), mini_batch[1][x].reshape(-1, 1))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #Atualizando os vetores de bias e as matrizes de pesos, somando-os(as) com o negativo dos seus respectivos gradientes
        #multiplicados por eta (coeficiente de aprendizagem), dividido pelo tamanho do mini_batch. OBS: Desta forma, utiliza-se
        #um gradiente não tão preciso pois só é baseado numa quantidade limitada de dados de treino (mini_batch), porém
        # exige-se um custo computacional bem menor
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def sigmoid(self,z):
        #retorna a função sigmoid (ativação do neurônio) tendo como entrada o z
        return (1.0 / (1.0 + np.exp(-z)))

#------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def sigmoid_derivate(self,z):
        #Derivada da função sigmoid
        return self.sigmoid(z)*(1-self.sigmoid(z))

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
    def prm_network (self, pesos, biases, sizes):
        #Passa parâmetros de uma rede neural
        self.weights = pesos
        self.biases = biases
        self.sizes = sizes

