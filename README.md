# IA_digitos_manuais
IA que entende dígitos manuais

Descrição breve de cada arquivo:

--------------------------------------------------------------------------------------------------

dados_testing.pkl.gz
Uma lista com a leitura pixel a pixel de imagens de dígitos de 0 a 9, contendo também a lista 
que descreve qual o dígito que cada imagem representa. Contém 10 mil imagens e tem como objetivo
o treino/teste da rede neural

dados_training.pkl.gz
Uma lista com a leitura pixel a pixel de imagens de dígitos de 0 a 9, contendo também a lista 
que descreve qual o dígito que cada imagem representa. Contém 60 mil imagens e tem como objetivo
o treino/teste da rede neural

rede_neural_90.pkl.gz
As matrizes de pesos, bias e sizes de uma rede neural com média de 90% de acerto na identificação
de dígitos manuais

--------------------------------------------------------------------------------------------------

main.py
Usado preferencialmente para chamar as funções do network.py

teste.py
Usado preferencialmente para chamar as funções do Network.output.py

network.py
Usado para criar e treinar as redes neurais

Network.output.py
Usado para testar as redes neurais

mnist_data.py
Usado para criar e formatar os dados que serão lidos pelas redes neurais como inputs e referencia.

--------------------------------------------------------------------------------------------------

data.rar
Contém a pasta data, depois mnist_png, que dentro, possui mais 2 pastas, a testing e a training. A pasta testing
contém 10 mil imagens de dígitos manuais, separadas em 10 pastas com o nome do dígito. A pasta
training é estruturada da mesma forma, porém, possui 60 mil imagens de dígitos no total
