'''
    Este modulo lê um arquivo NPZ que contenha faces e labels.
    Utiliza uma implementação pré-treinada da FaceNET para extrair
    características das faces.
    A saída é um arquivo NPZ que contem os features_vectors extraídos e
    as labels em formato numpy.
'''
#%%
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras as K

#%%
def extrair_caracteristicas(modelo, face):
    # Preparando imagem para o modelo:
    face =  face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean)/std # Normalizando os valores dos pixels
    samples = np.expand_dims(face, axis=0) 

    # Extraindo caracteristicas
    feat_vec = modelo.predict(samples)

    return feat_vec

#%%
if __name__ == '__main__':
    url_npz_in = 'resultados/5Celebs.npz'
    url_npz_out = 'resultados/5Celeb_FeatVec.npz'
    '''
        Recebe um NPZ que contenha imagens e labels e retorna um NPZ que
        contem vetores de características e labels.
        Entrada: NPZ
        Saída: NPZ
    '''
    # Lendo as faces detectadas e suas labels
    data = np.load(url_npz_in, allow_pickle=True)
    X_train = data['arr_0']
    y_train = data['arr_1']
    X_valid = data['arr_2']
    y_valid = data['arr_3']

    # Carregando o modelo FaceNet
    facenet = load_model('kerasFacenet/keras-facenet/model/facenet_keras.h5')

    # Extraindo caracteristicas das imagens de treino
    feat_vec_train = np.asarray([extrair_caracteristicas(facenet, face) for face in X_train])

    # Extraindo caracteristicas das imagens de validação
    feat_vec_valid = np.asarray([extrair_caracteristicas(facenet, face) for face in X_valid])

    # Salvando os vetores de características:
    np.savez_compressed(url_npz_out, X_train, y_train, X_valid, y_valid)
