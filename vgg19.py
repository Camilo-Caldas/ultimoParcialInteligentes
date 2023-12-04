from keras.applications import VGG19
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

def resumen(model=None):
    '''
    '''
    header = '{:4} {:16} {:24} {:24} {:10}'.format('#', 'Layer Name','Layer Input Shape','Layer Output Shape','Parameters'
    )
    print('='*(len(header)))
    print(header)
    print('='*(len(header)))
    count=0
    count_trainable=0
    for i, layer in enumerate(model.layers):
        count_trainable += layer.count_params() if layer.trainable else 0
        input_shape = '{}'.format(layer.input_shape)
        output_shape = '{}'.format(layer.output_shape)
        _str = '{:<4d} {:16} {:24} {:24} {:10}'.format(i,layer.name, input_shape, output_shape, layer.count_params())
        print(_str)
        count += layer.count_params()
    print('_'*(len(header)))
    print('Total Parameters : ', count)
    print('Total Trainable Parameters : ', count_trainable)
    print('Total No-Trainable Parameters : ', count-count_trainable)
    
  
vgg19=None

vgg19 = VGG19(weights='imagenet',
                  include_top=True,
                  input_shape=(224, 224, 3))

resumen(vgg19)

if vgg19 != None:
    del vgg19

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(256, 256, 3))
    
resumen(conv_base)    
conv_base.trainable = False

model = Sequential()
model.add(conv_base)        # modelo base agradado como una capa!
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='softmax'))

resumen(model)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=300)

ruta = "models/modeloVGG19.h5"
model.save(ruta)
model.summary()

imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas("dataset/test/", nombreCategorias,
                                                                      cantidaDatosPruebas, cantidaDatosEntrenamiento, ancho, alto)



resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)

# Evaluar el modelo en datos de prueba y mostrar métricas
print("Accuracy=",resultados[1])