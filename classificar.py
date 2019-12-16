'''
    Este modulo lê as features vectors extraídas e utiliza uma SVM para
    classificar e reconhecer os rostos
'''
#%%
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold

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

def teste_skf(path):
    data = np.load(path, allow_pickle=1)
    X, y = data['arr_0'].reshape(-1,128), data['arr_1']
    skf = StratifiedKFold(n_splits=5)
    svc = LinearSVC()
    acuracia = []
    precisao = []
    f1_lista = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        ac = accuracy_score(y_test, y_pred)
        acuracia.append(ac)
        print(f'Acurácia: {ac}')

        pre = precision_score(y_test, y_pred, average='macro')
        precisao.append(pre)
        print(f'Precisão: {pre}')

        f1 = f1_score(y_test, y_pred, average='macro')
        f1_lista.append(f1)
        print(f'F1 Score: {f1}')

        print('')

    print(f'Acurácia média: {np.array(acuracia).mean()}')
    print(f'Precisão média: {np.array(precisao).mean()}')
    print(f'F1 Score médio: {np.array(f1_lista).mean()}')
