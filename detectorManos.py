import os
import imageio
import cv2
import math
import numpy as np
import mediapipe as mp

TAM = 300
FRAMES_MIN = 5

def distancia(p1,p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def vectorizar(mano):
    base = mano.landmark[0]
    datos = []
    for lm in mano.landmark:
        datos.extend([lm.x - base.x, lm.y - base.y])
    datos = np.array(datos)
    return datos / np.max(np.abs(datos))

def contar_dedos(mano):
    dedos = []
    if mano.landmark[4].y < mano.landmark[3].y and mano.landmark[4].y < mano.landmark[5].y:
        dedos.append(1)
    else: 
        dedos.append(0)
    for punta in [8, 12, 16, 20]:
        if mano.landmark[punta].y < mano.landmark[punta - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)
    return dedos

def  cargar(ruta):
    foto = cv2.imread(ruta)
    if foto is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde la ruta: {ruta}")
    return cv2.resize(foto, (TAM, TAM))

def detectar_gesto(mano, dedos):
    if (distancia(mano.landmark[4], mano.landmark[8]) < 0.2 and
        distancia(mano.landmark[4], mano.landmark[12]) < 0.2 and
        distancia(mano.landmark[8], mano.landmark[12]) < 0.2 and
        dedos[2] == 1 and dedos[3] == 1):
        return "purpura"
    if  dedos == [0, 1, 1, 0, 0]:
        return "gato"
    if dedos == [0, 1, 0, 0, 0]:
        return "indice"
    if dedos == [1, 0, 0, 0, 0]:
        return "pulgarArriba"
    if dedos == [0, 0, 0, 0, 0]:
        return "puno"
    return None
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
fotos = {
    "gato": (cargar("Imagenes de prueba/gato.jpg"), "Gato"),
    "indice": (cargar("Imagenes de prueba/indice.jpg"), "Indice"),
    "pulgarArriba": (cargar("Imagenes de prueba/pulgarArriba.jpg"), "Pulgar Arriba"),
    "puno": (cargar("Imagenes de prueba/puno.jpg"), "Puno")
}

frames_gojo = imageio.mimread("Imagenes de prueba/purpura.gif", memtest=False)
frames_gojo = [cv2.resize(cv2.cvtColor(f[:, :, :3], cv2.COLOR_RGB2BGR), (300, 300)) for f in frames_gojo]
indice_gif = 0

if os.path.exists("muestras.npy"):
    muestras = list(np.load("muestras.npy"))
    etiquetas = list(np.load("etiquetas.npy"))
else:
    muestras = []
    etiquetas = []

gesto_actual = None
contador = 0
while True:
    success,img = dispositivo_Captura.read()
    
    if not success:
        break
    
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = manos.process(imgRGB)

    gesto = None 

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            mpDibujar.draw_landmarks(img, mano, mpManos.HAND_CONNECTIONS)
        
            #apartado para guardar poses de mano con las teclas 1, 2 y 0    
            if tecla == ord('1'):
                muestras.append(vectorizar(mano))
                etiquetas.append("pose1")
            if tecla == ord('2'):
                muestras.append(vectorizar(mano))
                etiquetas.append("pose2")
            if tecla == ord('0'):
                muestras.append(vectorizar(mano))
                etiquetas.append("nada")

            if gesto is None:
                gesto = detectar_gesto(mano, contar_dedos(mano))
    
    if gesto == gesto_actual:
        contador += 1
    else:
        contador = 0
        gesto_actual = gesto

    if gesto_actual is not None and contador >= FRAMES_MIN:
        if gesto_actual == "purpura":
            imagen = frames_gojo[indice_gif]
            indice_gif = (indice_gif + 1) % len(frames_gojo)
            nombre = "Purpura Hueco"
        else:
            imagen, nombre = fotos[gesto_actual] 
            indice_gif = 0

        img[0:TAM, 0:TAM] = imagen
        cv2.putText(img, nombre, (10, TAM + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   
    
    cv2.putText(img, f"muestras: {len(muestras)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("grabador", img)
    cv2.waitKey(1)

# al salir con q, se guardan las muestras
np.save("muestras.npy", np.array(muestras))
np.save("etiquetas.npy", np.array(etiquetas))
dispositivo_Captura.release()
cv2.destroyAllWindows()
