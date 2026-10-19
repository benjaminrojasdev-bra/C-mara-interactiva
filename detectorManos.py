import cv2
import mediapipe as mp
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

while True:
    success,img = dispositivo_Captura.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = manos.process(imgRGB)

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            #mpDibujar.draw_landmarks(img, mano, mpManos.HAND_CONNECTIONS)
            for id, lm in enumerate(mano.landmark):
                alto, ancho, color = img.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                if  id == 4:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                if  id == 20:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                
    #cambiar el color de la imagen a RGB
    cv2.imshow("image", img)
    cv2.waitKey(1)

