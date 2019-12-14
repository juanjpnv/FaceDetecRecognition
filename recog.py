#%%
import numpy as np
import cv2
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt
from IPython.display import Image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#%% Construindo a CNN
'''
    O modelo possuirá 8 camadas: convolutional, max pooling, convolutional, max pooling, convolutional, max pooling, convolucional, convolucional
    Fonte: https://ieeexplore.ieee.org/abstract/document/8248937

    => Na primeira camada adicionada é preciso determinar
        Formato da matriz de entrada: input_shape=(64,64,3)

    => Camada de Convolução:
        Quantidade de Kernels (numero de filtros): filters=32
        Dimensao do Kernel (dimensao do filtro): kernel_size=(3,3)
        Função de ativação: activation='relu'
            Uma Rede Neural sem função de ativação vira um modelo linear. A função de
            ativação mais usadas para imagens é a 'ReLU'
        Tamanho do deslocamento do deslizamento do kernel: stride=(1,1) - padrão do Keras

    => Camada Pooling: O objetido desta camada é diminuir a dimensionalidade do modelo e das features reduzindo a variância e a quantidade de parâmetros a serem treinados. O tipo de pooling: MaxPooling, SumPooling, AveregedPooling. O MaxPooling é o mais comum
            Tamanho do kernel de pooling: pool_size=(2,2)

    => Camada Totalmente Conectada (Dense ou Fully Connected): É uma rede neural tradicional
            As funções de ativação mais comuns são a softmax e a sigmoid: activation='softmax'


    Camadas opcionais para uma CNN: Devem vir antes da camada Totalmente Conectada
        Camada Dropout: ajuda a diminuar a sensibilidade ao ruído
                        Recebe como hiperparâmetro a possibilidade de desligar outras camadas durante o treino
        Camada Faltten: Utilizada para transformar as imagens (matrizes) em vetores de caracteristicas
'''

cnn =  Sequential() # Inicializando o modelo
cnn.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu') ) # adicionando camada
cnn.add(MaxPool2D(pool_size=(2,2))) # adicionando camada
cnn.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu') ) # adicionando camada
cnn.add(MaxPool2D(pool_size=(2,2))) # adicionando camada
cnn.add(Dropout(0.25)) # adicionando camada
cnn.add(Flatten()) # adicionando camada
cnn.add(Dense(128, activation='relu')) # adicionando camada
cnn.add(Dense(5, activation='softmax')) # adicionando camada

optimizer = Adam() # preparando otimizador
cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # compilando a rede
print(cnn.summary()) # imprimindo informações gerais sobre a rede

#%% Preparando base de dados
# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
datagen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             validation_split=0.2)

# Pré-processamento das imagens de treino e validação
training_set = datagen.flow_from_directory('/home/juan/Documentos/Scripts/FaceRecLBP/Datasets/5Celeb/Revisadas/Estrurutado',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            subset='training')

validation_set = datagen.flow_from_directory('/home/juan/Documentos/Scripts/FaceRecLBP/Datasets/5Celeb/Revisadas/Estrurutado',
                                             target_size = (64, 64),
                                             batch_size = 32,
                                             class_mode = 'categorical',
                                             subset='validation')

#%% Executando o treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
cnn.fit_generator(training_set,
                  steps_per_epoch = 100,
                  epochs = 5,
                  validation_data = validation_set,
                  validation_steps = 16)

#%% Executando testes

test_image = image.load_img('teste2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

labels = list(training_set.class_indices)

resposta = np.where(result)[1][0]


print(labels[resposta])

Image(filename='teste2.jpg')

# %%
