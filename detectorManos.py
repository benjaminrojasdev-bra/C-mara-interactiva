import cv2
import math
import mediapipe as mp

def distancia(p1,p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

#numero de la camara a utilizar: 0 porque es la de la camara del notebook
dispositivo_Captura = cv2.VideoCapture(0)

#importacion de los puntos de las manos
mpManos = mp.solutions.hands

#ocupacion de imagenes falsa
manos = mpManos.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.9,
                      min_tracking_confidence=0.8)

#importar utilidades de dibujo
mpDibujar = mp.solutions.drawing_utils

foto1 = cv2.imread("gato.jpg")
foto1 = cv2.resize(foto1, (200, 200))

foto2 = cv2.imread("indice.jpg")
foto2 = cv2.resize(foto2, (200, 200))

foto3 = cv2.imread("pulgarArriba.jpg")
foto3 = cv2.resize(foto3, (200, 200))

foto4 = cv2.imread("puno.jpg")
foto4 = cv2.resize(foto4, (200, 200))

foto5 = cv2.imread("wallace.jpg")
foto5 = cv2.resize(foto5, (200, 200))

foto6 = cv2.imread("gojo.jpg")
foto6 = cv2.resize(foto6, (200, 200))

while True:
    success,img = dispositivo_Captura.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = manos.process(imgRGB)

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            mpDibujar.draw_landmarks(img, mano, mpManos.HAND_CONNECTIONS)
            # aquí va la lógica de los dedos
        
        #if pulgar esta arriba
        
        
        puntas = [8, 12, 16, 20] # pulgar, indice, medio anular, meñique
        dedos = []
        
        if mano.landmark[4].y < mano.landmark[3].y and mano.landmark[4].y < mano.landmark[5].y:
            dedos.append(1)
        else:
            dedos.append(0)
        
        for punta in puntas:
            if mano.landmark[punta].y < mano.landmark[punta - 2].y:
                dedos.append(1)
            else:
                dedos.append(0)

        # aquí va el if que muestra la foto
        if dedos == [0, 1, 1, 0, 0]:
            img[0:200, 0:200] = foto1
        

        if dedos == [0, 1, 0, 0, 0]:
            img[0:200, 0:200] = foto2

        if dedos == [1, 0, 0, 0, 0]:
            img[0:200, 0:200] = foto3

        if dedos == [0, 0, 0, 0, 0]:
            img[0:200, 0:200] = foto4

        if distancia(mano.landmark[4], mano.landmark[8]) < 0.05 and dedos[1] == 1 and dedos[2] == 1 and dedos[3] == 1:
            img[0:200, 0:200] = foto5

        if distancia(mano.landmark[4], mano.landmark[8]) < 0.05 and distancia(mano.landmark[4], mano.landmark[12]) < 0.05 and distancia(mano.landmark[8], mano.landmark[12]) < 0.2 and dedos[2] == 1 and dedos[3] == 1:
            img[0:200, 0:200] = foto6
    #cambiar el color de la imagen a RGB
    cv2.imshow("image", img)
    cv2.waitKey(1)

