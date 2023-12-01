from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
import cargaData

ancho=256
alto=256
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
nombreCategorias= ['COVID','Lung Opacity','Normal','Viral Pneumonia']

#configuracion de las imagenes, en este caso 60 de entrenamiento y 20 de pruebas (para el número de categorías)
# cantidaDatosEntrenamiento=[6996,6571,6571,4292] 
# cantidaDatosPruebas=[8745,8214,8214,5365]

cantidaDatosEntrenamiento=[3000,3000,3000,3000] 
cantidaDatosPruebas=[4000,4000,4000,4000]

#Cargar las imágenes
imagenes, probabilidades= cargaData.cargar_datos("dataset/train/",nombreCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential() # porque es una capa despúes de la otra
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,))) # 784 pixeles correspondientes al vector aplanado de 28x28
model.add(Reshape(formaImagen)) # Al modelo se le agrega una reforma, es decir armar otra vez la matriz que fue aplanada

#Capas Ocultas
#Capas convolucionales, donde se va a experimentar en el parcial cambiando los parámetros
# Kernel size es arbitrario, implica que entre más grande reduce más, y entre más pequeño la imagen se demora más reduciendo (matriz roja de la presentacion)
# Strides, son los pasos
# Filtros es un valor arbitrario
# Padding, cuando se va deslizando el filtro lo que hace es que al estar al borde y estar parte de la matriz nula, la llena con los mismos valores anteriores
# Activation, se puede buscar en keras activations functions en la página de keras (Cambiar para el parcial para experimentar)
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1")) # capa convulocional 2D

# En la presentación pooling, strides se corre dos casillas, pool_size tamaño
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
# En sesta capa se haria el modelo mixto, para juntarlo con los datos por ejemplo de un dataframe
model.add(Flatten()) # parte final donde los datos que vienen en forma de matriz los aplana (antes de la capa output)
model.add(Dense(128,activation="relu")) # Agregar capa densa de 128 pixxeles

#Capa de salida
# Al modelo agreguele una capa densa que al final tenga 10 neuronas (Categorias), y poner los valores en terminos de 0 y 1
model.add(Dense(len(nombreCategorias),activation="softmax"))


#Traducir de keras a tensorflow
#Categorical_crossentropy cuando son de 3 categorías para arriba
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

#Entrenamiento
model.fit(x=imagenes,y=probabilidades,epochs=20,batch_size=150)

#Prueba del modelo
imagenesPrueba,probabilidadesPrueba= cargaData.cargar_datos_pruebas("dataset/test/",nombreCategorias,cantidaDatosPruebas, cantidaDatosEntrenamiento,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])

# Guardar modelo
ruta="models/modeloA.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()