# 🖐️ Cámara Interactiva

Programa en Python que detecta tus manos con la cámara del computador y muestra una imagen según el gesto que hagas.

## ✨ Gestos

| Gesto | Cómo se hace | Imagen |
|---|---|---|
| ✌️ Paz | Índice y medio arriba, resto abajo | gato |
| ☝️ Índice | Solo el índice arriba | indice |
| 👍 Pulgar arriba | Solo el pulgar arriba | pulgarArriba |
| ✊ Puño | Todos los dedos abajo | puno |
| 🤞 Pulgar con índice y medio | Pulgar tocando índice y medio, resto arriba | gojo |

## 🛠️ Tecnologías

- Python 3.12
- OpenCV
- MediaPipe 0.10.14
- NumPy
- imageio (para el GIF animado)

## 🚀 Cómo usarlo

1. Instala las librerías:
   pip install -r requirements.txt
2. Ejecuta el programa:
   python detectorManos.py
3. Haz los gestos frente a la cámara.
4. Presiona **q** (con la ventana de la cámara en foco) para salir.

## 🎥 Grabar tus propias poses

Con la cámara abierta, haz una pose y mantén apretada la tecla **1** o **2** para guardar muestras. La tecla **0** guarda "nada" (mano relajada). Al salir con **q** se guardan en `muestras.npy` y `etiquetas.npy`.

## 📌 Por hacer

- [x] Detección de dedos y distancias
- [x] Grabación de poses
- [x] GIF animado para la pose de Gojo
- [ ] Imagen más grande y estable
- [ ] Reconocimiento de poses grabadas
- [ ] Soporte de dos manos
