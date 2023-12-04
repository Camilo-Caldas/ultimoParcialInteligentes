#from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
#import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
import cargaData
from keras.applications.vgg19 import VGG19
from keras.models import Model


from keras.preprocessing.image import ImageDataGenerator

ancho=256
alto=256
IMAGE_SIZE = [ancho, alto]
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=3
formaImagen=(ancho,alto,numeroCanales)
nombreCategorias= ['COVID','Lung Opacity','Normal','Viral Pneumonia']

#configuracion de las imagenes, en este caso 60 de entrenamiento y 20 de pruebas (para el número de categorías)
cantidaDatosEntrenamiento=[6996,6571,6571,4292] 
cantidaDatosPruebas=[8745,7674,8214,5365]

#Cargar las imágenes
imagenes, probabilidades= cargaData.cargar_datos("dataset/train/",nombreCategorias,cantidaDatosEntrenamiento,ancho,alto)

#modelo VGG19
# Create a VGG16 model, and removing the last layer that is classifying 1000 images. This will be replaced with images classes we have. 
# Crea el modelo VGG16 con la forma de entrada adecuada
# model = Sequential()
# model.add(Dense(256, 256, 3, input_shape=(256, 256, 3)))
# model.add(VGG19(include_top=False))
# model.add(VGG19(include_top=False, input_shape=(256, 256, 3)))
# model.add(Flatten())  # Aplana la salida de la capa VGG16
# Agrega capas adicionales según sea necesario
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

vgg19 = keras.applications.vgg19
conv_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in conv_model.layers: 
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)
model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
model.summary()

# Compila y entrena el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x=imagenes, y=probabilidades, epochs=20, batch_size=800)

#Prueba del modelo
imagenesPrueba,probabilidadesPrueba= cargaData.cargar_datos_pruebas("dataset/test/",nombreCategorias,cantidaDatosPruebas, cantidaDatosEntrenamiento,ancho,alto)



# Crear generadores de imágenes
datagen_entrenamiento = ImageDataGenerator(rescale=1./255)  # Puedes ajustar esto según tus necesidades
generador_entrenamiento = datagen_entrenamiento.flow_from_directory(
    "dataset/train/",
    target_size=(224, 224),
    batch_size=800,  # Ajusta según tus necesidades
    class_mode='categorical'  # Ajusta según tus necesidades
)

datagen_prueba = ImageDataGenerator(rescale=1./255)  # Puedes ajustar esto según tus necesidades
generador_prueba = datagen_prueba.flow_from_directory(
    "dataset/test/",
    target_size=(224, 224),
    batch_size=800,  # Ajusta según tus necesidades
    class_mode='categorical'  # Ajusta según tus necesidades
)
model.fit_generator(
    generator=generador_entrenamiento,
    validation_data=generador_prueba,
    epochs=5,
    workers=10
)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])


# Guardar modelo
ruta="models/modeloVGG19.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()



