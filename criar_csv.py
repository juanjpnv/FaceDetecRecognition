import os
import numpy as np
import pandas as pd

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
def tudo_em_uma_pasta(path):
    '''
        Nesse caso, a label é parte do nome da imagem
    '''
    separador = ","
    print(f'Path{separador}Label')
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            abs_path = f'{path}/{filename}'
            print(f'{abs_path}{separador}{filename[:-7]}')

def separados_por_pastas(path):
    '''
        adaptado de philipp@mango:~/facerec/data/at$ tree

        Ler pastas com a seguinte estrutura:
        |-- path
        |   |-- s1
        |   |   |-- item1.extensao
        |   |   |-- ...
        |   |   |-- item10.extensao
        |   |-- s2
        |   |   |-- item1.extensao
        |   |   |-- ...
        |   |   |-- item10.extensao
        |   |-- ...
        |   |-- s40
        |   |   |-- item1.extensao
        |   |   |-- ...
        |   |   |-- item10.extensao
        
        E retorna uma lista com os caminhos de cada item.
        A label é o nome da pasta de cada imagem
    '''
    separador = ","
    lista = list()
    print(f'Path{separador}Label')
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = f'{subject_path}/{filename}'
                linha = f'{abs_path}{separador}{subdirname}'
                lista.append(linha)
                print(f'{abs_path}{separador}{subdirname}')
    return lista

def gerar_csv(lista, path):
    data = list()
    for item in lista:
        data.append(item.split(','))

    data = np.array(data)
    dataframe = pd.DataFrame(list(zip(data[:,0],data[:,1])),columns=['url','label'])

    dataframe.to_csv(path, index=0)

tipo = input("Cada pessoa tem pasta propria (S/N): ")
path = input("Path: ")

if (tipo == "S"):
    lista = separados_por_pastas(path)
    gerar_csv(lista, 'arquivoCSV.csv')
elif (tipo == "N"):
    tudo_em_uma_pasta(path)
else:
    print('Tente depois')