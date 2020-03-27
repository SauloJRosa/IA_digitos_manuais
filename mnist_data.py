from PIL import Image
import numpy as np
import pickle
import gzip
import shutil
import os

#----------------------------------------------------------------------------------------------#
''' Esta função recebe um parâmetro w, que significará quantas imagens de cada digito ele fará 
o processamento da imagem de forma a gerar um vetor_imagem contendo a escala em cinza dos pixels de
cada imagem 28x28 juntamente com um vetor de referência contendo qual digito a imagem equivale '''

def processar_dados():

    '''#10 mil imagens de training
    quantidade = {
        0: 980,
        1: 1135,
        2: 1032,
        3: 1010,
        4: 982,
        5: 892,
        6: 958,
        7: 1028,
        8: 974,
        9: 1009
    }
    '''

    #60 mil imagens de testing
    quantidade = {
        0: 5923,
        1: 6742,
        2: 5958,
        3: 6131,
        4: 5842,
        5: 5421,
        6: 5918,
        7: 6265,
        8: 5851,
        9: 5949
    }

    n = 0
    m = 1

    vetor_resposta = []
    vetor_imagem = []

    while (n<10):

        while (m<=quantidade.get(n)):

            vetor_imagem.append(map_pixels(n,m))
            vetor_resposta.append(referencia(n))
            m += 1

        n += 1
        m = 1

    vetor_dados = [vetor_imagem , vetor_resposta]
    vetor_dados = np.asarray(vetor_dados)

    pickle_out = open("dados_training.pkl","wb")
    pickle.dump(vetor_dados,pickle_out)
    pickle_out.close()

    with open('dados_training.pkl','rb') as f_input:
        with gzip.open('dados_training.pkl.gz','wb') as f_output:
            shutil.copyfileobj(f_input,f_output)

    os.remove('dados_training.pkl')

#----------------------------------------------------------------------------------------------#
''' Função que mapeia todos os pixels de uma imagem 28x28 e aloca as informações da escala de
 cinza em um vetor ver_img'''

def map_pixels(n,m):

    vet_img =[]
    # coleta da imagem que será analisada
    im = Image.open(
        'coloque o endereço da pasta dos arquivos de imagem aqui/training/{}/{} ({}).png'.format(n, n, m))
    # print(im.format, im.size, im.mode)
    px = im.load()

    # Loop que varre a imagem e cria o vetor de pixels
    x = y = 0
    while x < 28:
        while y < 28:
            vet_img.append(round((px[y, x] / 255), 2))
            y += 1
        x += 1
        y = 0

    vet_img = np.asarray(vet_img)
    return (vet_img)

#----------------------------------------------------------------------------------------------#
''' Função que retorna o valor de referência através de um vetor de 10 espaços. Ele monta o vetor
com base no índice n, que é usado na chamada da função'''

def referencia(n):

    vet_resp = []
    # vetor de resposta correta do digito escrito
    t = 0
    while t < 10:
        if t == n:
            vet_resp.append(1)
        if t == 9:
            break
        vet_resp.append(0)
        t += 1

    vet_resp = np.asarray(vet_resp)
    return (vet_resp)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
def carregar_dados_training():
    #Abre o zip e transfere os dados de todas as imagens (60 mil imagens no caso
    # de 'dados.pkl.gz' ou 10 mil imagens no caso de 'dados_testing.pkl.gz') e respostas para o vetor_dados
    with gzip.open('dados_training.pkl.gz', 'rb') as f:
        vetor_dados = pickle.load(f)

    return vetor_dados

# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
def carregar_dados_testing():
    #Abre o zip e transfere os dados de todas as imagens (60 mil imagens no caso
    # de 'dados.pkl.gz' ou 10 mil imagens no caso de 'dados_testing.pkl.gz') e respostas para o vetor_dados
    with gzip.open('dados_testing.pkl.gz', 'rb') as f:
        vetor_dados = pickle.load(f)

    return vetor_dados

# ------------------------------------------------------------------------------------------------------------------------------------------------------------#
