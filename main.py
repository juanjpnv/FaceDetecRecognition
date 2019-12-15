'''
    Este módulo integra o detector, extrator e o classificador
'''

#%%
import cv2
import numpy as np
from classificar import Reconhecer
from feature_extractor import ExtairCaracteristicas

#%%
def recortar(x1, y1, w, h, img):
    X1, Y1 = int(x1), int(y1) 
    X2, Y2 = int(x1+w), int(y1+h)

    cara = img[Y1:Y2, X1:X2] # recorta o rosto
    cv2.rectangle(img, (X1, Y1), (X2, Y2), (255, 0 , 0))
    return cara

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
    # bounding box
    for (x, y, w, h) in faces:          
        # desenhando bb e recortando rosto
        X1, Y1 = int(x), int(y) 
        X2, Y2 = int(x+w), int(y+h)

        rosto = frame[Y1:Y2, X1:X2] # recorta o rosto
        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0 , 0))
        
        rosto = EC.preprocessamento(rosto)
        ftvec = EC.extrair_caracteristicas(rosto)

        pessoa = reconhecedor.prever(ftvec)
        cv2.putText(frame, f'{pessoa[0]}',(X1,Y2+10), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    
    cv2.imshow("Video", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
captura.release()
cv2.destroyAllWindows()

# %%
