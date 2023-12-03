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
cantidaDatosEntrenamiento=[6996,6571,6571,4292] 
cantidaDatosPruebas=[8745,7674,8214,5365]

#Cargar las imágenes
imagenes, probabilidades= cargaData.cargar_datos("dataset/train/",nombreCategorias,cantidaDatosEntrenamiento,ancho,alto)

#modelo VGG19
tf.keras.applications.vgg19.VGG19(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)