import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten

def cargar_datos(ruta_origen,nombre_categorias,limite,ancho,alto):
    imagenes_cargadas=[]
    valor_esperado=[]
    index = 0
    for categoria in nombre_categorias:
        for id_imagen in range(0,limite[index]):
            ruta=ruta_origen+str(categoria)+"/"+str(id_imagen)+".png"
            #print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # convierte la imagen a escala de grises
            imagen = cv2.resize(imagen, (ancho, alto)) # Redimensiona la imagen al ancho y alto configurado
            imagen = imagen.flatten() # Pasa la matriz a un vector
            imagen = imagen / 255 # Normalización (Se pueden utilizar otras)
            imagenes_cargadas.append(imagen) # A las imagenes cargadas se le agrega la imagen aplanada
            probabilidades = np.zeros(len(nombre_categorias))
            probabilidades[index] = 1
            valor_esperado.append(probabilidades)
        index = index + 1
    imagenes_entrenamiento = np.array(imagenes_cargadas)
    valores_esperados = np.array(valor_esperado)
    return imagenes_entrenamiento, valores_esperados

def cargar_datos_pruebas(ruta_origen,nombre_categorias,limite, limite_inferior,ancho,alto):
    imagenes_cargadas=[]
    valor_esperado=[]
    index = 0
    for categoria in nombre_categorias:
        for id_imagen in range(limite_inferior[index],limite[index]):
            ruta=ruta_origen+str(categoria)+"/"+str(id_imagen)+".png"
            #print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # convierte la imagen a escala de grises
            imagen = cv2.resize(imagen, (ancho, alto)) # Redimensiona la imagen al ancho y alto configurado
            imagen = imagen.flatten() # Pasa la matriz a un vector
            imagen = imagen / 255 # Normalización (Se pueden utilizar otras)
            imagenes_cargadas.append(imagen) # A las imagenes cargadas se le agrega la imagen aplanada
            probabilidades = np.zeros(len(nombre_categorias))
            probabilidades[index] = 1
            valor_esperado.append(probabilidades)
        index = index + 1
    imagenes_entrenamiento = np.array(imagenes_cargadas)
    valores_esperados = np.array(valor_esperado)
    return imagenes_entrenamiento, valores_esperados


