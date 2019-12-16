'''
    Este módulo integra o detector, extrator e o classificador
'''

#%%
import cv2
import numpy as np
from classificar import Reconhecer
from feature_extractor import ExtairCaracteristicas
import detect

#%%
def recortar(x1, y1, w, h, img):
    X1, Y1 = int(x1), int(y1) 
    X2, Y2 = int(x1+w), int(y1+h)

    cara = img[Y1:Y2, X1:X2] # recorta o rosto
    cv2.rectangle(img, (X1, Y1), (X2, Y2), (255, 0 , 0))
    return cara

def novo_NPZ_de_features(path, nome):
    '''
        Ler um CSV com fotos, extrai os rostos e as caracteristicas.
        Gera um novos NPZ na pasta dataset
    '''
    detect.preparar_e_salvar_dataset(path, f'resultados/{nome}.npz')
    EC = ExtairCaracteristicas()
    EC.criar_feature_npz(f'resultados/{nome}.npz',f'resultados/{nome}_FeatVec.npz')

#%%
# inicializando extrator de características
EC = ExtairCaracteristicas()

# treinando o modelo
reconhecedor = Reconhecer('resultados/juaneantonia_FeatVec.npz')

# iniciando Haarcascade
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Inciando video webcam
captura = cv2.VideoCapture(0) 
while(1):
    ret, frame = captura.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # procurando rosto no vídeo com haar cascade
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    print(f'Rostos encontrados: {len(faces)}') # Imprimi a quantidade de rostos encontrados

    # bounding box
    for (x, y, w, h) in faces:          
        # desenhando bb e recortando rosto
        X1, Y1 = int(x), int(y) 
        X2, Y2 = int(x+w), int(y+h)

        rosto = frame[Y1:Y2, X1:X2] # recorta o rosto
        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0 , 0)) # desenha Boundingbox
        
        rosto = EC.preprocessamento(rosto) # processa rosto para extração de carcteristicas
        ftvec = EC.extrair_caracteristicas(rosto) # extrai as caracteristicas

        pessoa = reconhecedor.prever(ftvec) # "adivinha" que é
        cv2.rectangle(frame, (X1, Y2), (X2,Y2+20), (255, 0, 0), -1) # retangulo preenchido
        cv2.putText(frame, f'{pessoa[0]}',(X1+15,Y2+15), cv2.FONT_HERSHEY_PLAIN, 1, (255,250,250))

        print(f'{pessoa[0]}') # imprime o nome da pessoa
    print('')

    cv2.imshow("Video", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
captura.release()
cv2.destroyAllWindows()

# %%
