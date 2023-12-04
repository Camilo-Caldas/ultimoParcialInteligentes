# from keras.applications import VGG19
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import keras
# import numpy as np
# import cv2
# ###Importar componentes de la red neuronal
# from keras.models import Sequential, Model
# from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten,LeakyReLU, Dropout, GlobalMaxPooling2D
# import cargaData
# from sklearn.metrics import f1_score
# from keras.optimizers import Adagrad

from keras.layers import Input,Dense,Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from datetime import datetime
from keras.callbacks import ModelCheckpoint

ancho = 256
alto = 256
pixeles = ancho * alto
numeroCanales = 3
formaImagen = (ancho, alto, numeroCanales)
nombreCategorias = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

cantidaDatosEntrenamiento = [6996, 6571, 6571, 4292]
cantidaDatosPruebas = [8745, 7674, 8214, 5365]

#imagenes, probabilidades = cargaData.cargar_datos("dataset/train/", nombreCategorias, cantidaDatosEntrenamiento, ancho, alto)

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
    
  
# vgg19=None

# vgg19 = VGG19(weights='imagenet',
#                   include_top=True,
#                   input_shape=(224, 224, 3))

# resumen(vgg19)

# if vgg19 != None:
#     del vgg19

# setting image size 
IMAGE_SIZE = [ 256 , 256 , 3 ]

# Load the model 
vgg = VGG19( include_top = False,
            input_shape = IMAGE_SIZE,
            weights = 'imagenet')

# Visualize the model
vgg.summary()

for  layer in vgg.layers:
    layer.trainable = False

# Flattened the last layer
x = Flatten()(vgg.output)

# Created a new layer as output
prediction = Dense( len(nombreCategorias) , activation = 'softmax' )(x)

# Join it with the model
model = Model( inputs = vgg.input , outputs = prediction )

# Visualize the model again
model.summary()

# defining adam
adam=Adam()

# compining the model
model.compile( loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'] )

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input ,
    rotation_range = 40 ,
    width_shift_range = 0.2 ,
    height_shift_range = 0.2 ,
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True ,
    fill_mode = 'nearest'
)

# Doing similar for the test data also

test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input ,
    rotation_range = 40 ,
    width_shift_range = 0.2 ,
    height_shift_range = 0.2 ,
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True ,
    fill_mode = 'nearest'
)

train_path = 'dataset/train'
test_path = 'dataset/test'

# train data
train_set = train_datagen.flow_from_directory(train_path,
                                            target_size = ( 256 , 256 ),
                                            batch_size = 900,
                                            class_mode = 'categorical')

# test data
test_set = test_datagen.flow_from_directory(test_path,
                                             target_size = ( 256 , 256 ),
                                            batch_size = 900,
                                            class_mode = 'categorical')

model.fit( train_set,
                          validation_data = test_set,
                          epochs = 10,
                          steps_per_epoch = 5,
                          validation_steps = 32,
                          verbose = 2)

ruta = "models/modeloVGG19.h5"
model.save(ruta)
model.summary()
# resumen(conv_base)    
# conv_base.trainable = False


# model = Sequential()
# model.add(conv_base)        # modelo base agradado como una capa!
# model.add(Flatten())
# model.add(Reshape(formaImagen))
# model.add(Dense(128, activation='relu'))
# #model.add(Reshape(formaImagen))
# model.add(Dense(len(nombreCategorias), activation="softmax"))




# resumen(model)

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=300)

# width_shape = 256
# height_shape = 256
# num_classes = len(nombreCategorias)
# epochs = 50
# batch_size = 32 

# image_input = Input(shape=(width_shape, height_shape, numeroCanales))

# m_VGG19 = VGG19(input_tensor=image_input, include_top=False,weights='imagenet')

# m_VGG19.summary()

# last_layer = m_VGG19.layers[-1].output
# x= Flatten(name='flatten')(last_layer)
# x = Dense(10, activation='relu', name='fc1')(x)
# x=Dropout(0.3)(x)
# x = Dense(10, activation='relu', name='fc2')(x)
# x=Dropout(0.3)(x)
# out = Dense(num_classes, activation='softmax', name='output')(x)
# custom_model = Model(image_input, out)
# custom_model.summary()

# # freeze all the layers except the dense layers
# for layer in custom_model.layers[:-6]:
# 	layer.trainable = False

# custom_model.summary()

# custom_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# ruta = "models/modeloVGG19.h5"
# custom_model.save(ruta)
# custom_model.summary()

# imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas("dataset/test/", nombreCategorias,
#                                                                       cantidaDatosPruebas, cantidaDatosEntrenamiento, ancho, alto)



# resultados = custom_model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)

# # Evaluar el modelo en datos de prueba y mostrar métricas
# print("Accuracy=",resultados[1])