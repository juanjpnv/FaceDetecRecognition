'''
    Este modulo lê as features vectors extraídas e utiliza uma SVM para
    classificar e reconhecer os rostos
'''
#%%
import numpy as np
from sklearn.svm import LinearSVC

#%%
class Reconhecer:
    '''
        Recebe um NPZ com features_vectors e treina um modelo LinearSVC
    '''
    def __init__(self, url_npz):
        # iniciando os dados
        self.X_train, self.y_train, self.X_valid, self.y_valid = self.__ler_npz(url_npz)
        # iniciando o classificador
        self.svm = LinearSVC()
        # realizando o treinamento
        self.__treinar(self.X_train, self.y_train)

    def __treinar(self, X, y):
        self.svm.fit(X, y)
    
    def __ler_npz(self, path):
        data = np.load(path, allow_pickle=1)
        if len(data.files) == 2:
            X_train = data['arr_0'].reshape(-1,128)
            y_train = data['arr_1']
            return X_train, y_train, None, None

        # Extraindo caracteristicas das imagens de validação
        elif len(data.files) == 4:
            X_train = data['arr_0'].reshape(-1,128)
            y_train = data['arr_1']
            X_valid = data['arr_2'].reshape(-1,128)
            y_valid = data['arr_3']
            return X_train, y_train, X_valid, y_valid

    def prever(self, X):
        '''
            X = feature vector da imagem a ser prevista
        '''
        return self.svm.predict(X)