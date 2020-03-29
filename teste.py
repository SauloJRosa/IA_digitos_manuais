import mnist_data
import Network_output
import pickle
import gzip

#carregando um vetor de dados testing com 10 mil imagens
vetor_dados = mnist_data.carregar_dados_testing()

#Pegando as informações da rede_neural_90 e alocando em pesos, biases e sizes
with gzip.open('rede_neural_90.pkl.gz', 'rb') as f:
    pesos, biases, sizes = pickle.load(f)

#Número de testes aleatórios desejados
testes = 1000

#fornecendo as informações da rede neural para a classe network_output
net = Network_output.network_output(pesos, biases, sizes)

#Fazer os testes aleatórios no vetor de dados carregados
net.testes_aleat(vetor_dados, testes)


