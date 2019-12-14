'''
    Este módulo encontra as imagens e labels através de arquivo CSV.
    Utiliza uma implementação de MultiTask CNN para encontrar os rostos
    em uma imagem.
    A saída final é um arquivo NPZ que contem as imagens e labels em
    formato numpy
'''
#%%
import cv2
from mtcnn.mtcnn import MTCNN	
import numpy as np
import pandas as pd

#%%
# gerando instância de detecção facial
detector = MTCNN()

#%%
def extrair_face(url, output_size=(160,160)):
    '''
        Recebe o caminho de uma imagem, faz a leitura em RGB
        e retorna apenas as faces encontradas na imagem
    '''
    # Lendo imagem em RGB
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detectando faces em uma imagem
    faces = detector.detect_faces(img)

    rosto = list() # variável que guarda os rostos extraídos
    for face in faces:
        x1, y1, w, h = face['box'] # recebendo variaveis do bounding box
        x1, y1, w, h = np.abs(x1), np.abs(y1), np.abs(w), np.abs(h)

        # Definindo os pontos para recortar o rosto. Precisam ser inteiros.
        X1, Y1 = int(x1), int(y1) 
        X2, Y2 = int(x1+w), int(y1+h)

        cara = img[Y1:Y2, X1:X2] # recorta o rosto

        rosto.append(cv2.resize(cara, output_size)) # salva o rosto numa lista
    
    # O codigo encontra e recorta todos os rostos. Mas por enquanto,
    # esta função retorna apenas o primeiro rosto encontrado
    return np.asarray(rosto[0])

#%%
def preparar_dataset(csv_url):
    dataset = pd.read_csv(csv_url)
    urls, label = dataset.values[:, 0], dataset.values[:, 1]

    print('Processando...')

    faces = list(range(len(urls)))
    i = 0

    faces = [extrair_face(url) for url in urls]
    '''
    for url in urls:
        faces[i] = extrair_face(url)
        i = i + 1
    '''

    print(f'{len(urls)} imagens')
    print(f'{len(np.unique(label))} classes')

    return np.asarray(faces), np.asarray(label).reshape(-1,1)

#%% Salvando dados preparados
def preparar_e_salvar_dataset(train_url, valid_url=None):
    '''
        Essa função prepara um arquivo NPZ com os dados de treino e validação
    '''
    X_train, y_train = preparar_dataset(train_url)
    X_valid, y_valid = preparar_dataset(valid_url)

    np.savez_compressed('resultados/5Celebs.npz', X_train, y_train, X_valid, y_valid)

#%%
if __name__ == '__main__':
    train_url='datasets/5CelebTrain.csv'
    valid_url='datasets/5CelebValid.csv'

    preparar_e_salvar_dataset(train_url, valid_url)
