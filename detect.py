import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

file_name = 'test3.jpg'
pixels = cv2.imread(file_name)

detector = MTCNN()
faces = detector.detect_faces(pixels)

x = np.zeros(len(faces))
y = np.zeros(len(faces))
w = np.zeros(len(faces))
h = np.zeros(len(faces))
keypoints = list(range(len(faces)))

i = 0
for face in faces:
    x[i], y[i], w[i], h[i] = face['box']
    keypoints[i] = face['keypoints'].values()
    i = i + 1


img = cv2.imread(file_name)
for i in range(len(faces)):
    pt1 = (int(x[i]), int(y[i]))
    pt2 = (int(x[i]+w[i]), int(y[i]+h[i]))
    cor = (255, 0, 0)
    cv2.rectangle(img, pt1, pt2, cor)

    for kp in keypoints[i]:
        cv2.circle(img, kp, 1, cor,thickness=2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()