from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
import cargaData
from sklearn.metrics import f1_score

ancho = 256
alto = 256
pixeles = ancho * alto
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
nombreCategorias = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

cantidaDatosEntrenamiento = [6996, 6571, 6571, 4292]
cantidaDatosPruebas = [8745, 7674, 8214, 5365]

imagenes, probabilidades = cargaData.cargar_datos("dataset/train/", nombreCategorias, cantidaDatosEntrenamiento, ancho, alto)


# Creación del modelo secuencial
model = Sequential()

# Capa de entrada: Transformación del tensor de entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

# Primera capa convolucional seguida de una capa de MaxPooling
model.add(Conv2D(kernel_size=5, strides=2, filters=16, padding="same", activation="relu", name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

# Segunda capa convolucional seguida de otra capa de MaxPooling
model.add(Conv2D(kernel_size=3, strides=1, filters=36, padding="same", activation="relu", name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))

# Capas convolucionales adicionales
model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding="same", activation="relu", name="capa_3"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding="same", activation="relu", name="capa_4"))
model.add(MaxPool2D(pool_size=2, strides=2))

# Capa de aplanamiento (Flatten)
model.add(Flatten())

# Capas densas (total 3) con funciones de activación ReLU
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))

# Capa de salida con función de activación softmax
model.add(Dense(len(nombreCategorias), activation="softmax"))

# Compilación del modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento del modelo
model.fit(x=imagenes, y=probabilidades, epochs=20, batch_size=800)

imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas("dataset/test/", nombreCategorias,
                                                                      cantidaDatosPruebas, cantidaDatosEntrenamiento, ancho, alto)

resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)

# Evaluar el modelo en datos de prueba y mostrar métricas
print("Accuracy=",resultados[1])

ruta = "models/modeloDos.h5"
model.save(ruta)
model.summary()