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
class ExtairCaracteristicas:
    def __init__(self, local_modelo='kerasFacenet/keras-facenet/model/facenet_keras.h5'):
        self.facenet = load_model(local_modelo)

    def preprocessamento(self, img):
        '''
            Recebe uma imagem de rosto e preprocessa para que
            possa ter suas características extraídas
        '''
        rosto = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rosto = cv2.resize(rosto, (160,160))
        
        return rosto

    def extrair_caracteristicas(self, face):
        # Preparando imagem para o modelo:
        modelo = self.facenet
        face =  face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean)/std # Normalizando os valores dos pixels
        samples = np.expand_dims(face, axis=0) 

        # Extraindo caracteristicas
        feat_vec = modelo.predict(samples)

        return feat_vec

    def criar_feature_npz(self, url_npz_in, url_npz_out):
        '''
            Recebe um NPZ que contenha imagens e labels e retorna um NPZ que
            contem vetores de características e labels.
            Entrada: NPZ
            Saída: NPZ
        '''
        # Lendo as faces detectadas e suas labels
        data = np.load(url_npz_in, allow_pickle=True)

        
        if len(data.files) == 2:
            # Extraindo caracteristicas das imagens de treino
            X_train, y_train = data['arr_0'], data['arr_1']
            feat_vec_train = np.asarray([self.extrair_caracteristicas(face) for face in X_train])
            # Salvando os vetores de características:
            np.savez_compressed(url_npz_out, feat_vec_train, y_train)

        # Extraindo caracteristicas das imagens de validação
        elif len(data.files) == 4:
            # Extraindo caracteristicas das imagens de treino
            X_train, y_train = data['arr_0'], data['arr_1']
            feat_vec_train = np.asarray([self.extrair_caracteristicas(face) for face in X_train])
            # Extraindo caracteristicas das imagens de teste
            X_valid, y_valid = data['arr_2'], data['arr_3']
            feat_vec_valid = np.asarray([self.extrair_caracteristicas(face) for face in X_valid])
            # Salvando os vetores de características:
            np.savez_compressed(url_npz_out, feat_vec_train, y_train, feat_vec_valid, y_valid)


#%%
if __name__ == '__main__':
    extrator = ExtairCaracteristicas()

    url_in = 'resultados/juaneantonia.npz'
    url_out = 'resultados/juaneantonia_FeatVec.npz'
    extrator.criar_feature_npz(url_in, url_out)