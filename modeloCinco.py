from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten,LeakyReLU, Dropout, GlobalMaxPooling2D
import cargaData
from sklearn.metrics import f1_score
from keras.optimizers import Adagrad

ancho = 256
alto = 256
pixeles = ancho * alto
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
nombreCategorias = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

cantidaDatosEntrenamiento = [6996, 6571, 6571, 4292]
cantidaDatosPruebas = [8745, 7674, 8214, 5365]

imagenes, probabilidades = cargaData.cargar_datos("dataset/train/", nombreCategorias, cantidaDatosEntrenamiento, ancho, alto)

# Creación del nuevo modelo secuencial
model = Sequential()

# Capa de entrada: Transformación del tensor de entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

# Reduce the number of filters and kernel size
model.add(Conv2D(kernel_size=3, strides=1, filters=2, padding="same", activation="relu", name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=4, padding="same", activation="relu", name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=6, padding="same", activation="relu", name="capa_3"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=8, padding="same", activation="relu", name="capa_4"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=2, filters=10, padding="same", activation="relu", name="capa_5"))
model.add(MaxPool2D(pool_size=2, strides=1))

# Further reduce the complexity
model.add(Flatten())
model.add(Dense(128, activation="relu"))

# Output layer 
model.add(Dense(len(nombreCategorias), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x=imagenes, y=probabilidades, epochs=35, batch_size=100)

ruta = "models/modeloCinco.h5"
model.save(ruta)
model.summary()

imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas("dataset/test/", nombreCategorias,
                                                                      cantidaDatosPruebas, cantidaDatosEntrenamiento, ancho, alto)



resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)

# Evaluar el modelo en datos de prueba y mostrar métricas
print("Accuracy=",resultados[1])