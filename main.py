import os
import random
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Input, MaxPool2D, Flatten, Conv2D, Dense, Reshape
from keras.applications import VGG16, ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# def cargarDatos(rutaOrigen, numeroCategorias, limite, ancho, alto):
#     imagenesCargadas = []
#     valorEsperado = []
#     for categoria in range(0, numeroCategorias):
#         for idImagen in range(1, limite[categoria] + 1):  # Corregir el límite
#             ruta = f"{rutaOrigen}{categoria}/{categoria}_{idImagen}.jpg"
#             print(ruta)
#             imagen = cv2.imread(ruta)
#             imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#             imagen = cv2.resize(imagen, (ancho, alto))
#             imagen = imagen.flatten() / 255.0
#             imagenesCargadas.append(imagen)
#             probabilidades = np.zeros(numeroCategorias)
#             probabilidades[categoria] = 1
#             valorEsperado.append(probabilidades)
#     imagenesEntrenamiento = np.array(imagenesCargadas)
#     valoresEsperados = np.array(valorEsperado)
#     return imagenesEntrenamiento, valoresEsperados

# def build_lenet_model(input_shape, num_classes):
#     model = Sequential()
#     model.add(InputLayer(input_shape=input_shape))
#     model.add(Reshape((input_shape[0], input_shape[1], 1)))
#     model.add(Conv2D(6, kernel_size=5, activation='relu'))
#     model.add(MaxPool2D(pool_size=2, strides=2))
#     model.add(Conv2D(16, kernel_size=5, activation='relu'))
#     model.add(MaxPool2D(pool_size=2, strides=2))
#     model.add(Flatten())
#     model.add(Dense(120, activation='relu'))
#     model.add(Dense(84, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

# def build_vgg16_model(input_shape, num_classes):
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
#     model = Sequential()
#     model.add(base_model)
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

# def build_resnet50_model(input_shape, num_classes):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#     model = Sequential()
#     model.add(base_model)
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

# ancho = 256
# alto = 256
# pixeles = ancho * alto
# numeroCanales = 1
# formaImagen = (ancho, alto, numeroCanales)
# numeroCategorias = 2

# labeling was start

categories=["COVID-19","NON-COVID"]

labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)

ruta_absoluta = "D:/Estudio/Inteligentes 2/ultimoParcialInteligentes/dataset/train/COVID"
directorio_base = "D:/Estudio/Inteligentes 2/ultimoParcialInteligentes/"

ruta_relativa = os.path.relpath(ruta_absoluta, directorio_base)
print(ruta_relativa)

#"COVID-19"
folder_path_covid =[
    "dataset/train/COVID"
]

def file_add_to_array(folder_path,append_array):
    img_names=os.listdir(folder_path)
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        append_array.append(img_path)
    

img_path_covid_old =[]
for folder_path in folder_path_covid:
    file_add_to_array(folder_path,img_path_covid_old)
        

# "NON-COVID"
folder_path_non_covid =[
    "dataset/train/Normal",
    "dataset/train/Lung opacity",
    "dataset/train/Viral Pneumonia",
]


img_path_non_covid_old =[]
for folder_path in folder_path_non_covid:
    file_add_to_array(folder_path,img_path_non_covid_old)
    

covid_images_on_data_set = len(img_path_covid_old)
covid_non_images_on_data_set = len(img_path_non_covid_old)
Total_images_on_data_set = covid_images_on_data_set + covid_non_images_on_data_set

print(f" Total Images on the data set : {Total_images_on_data_set}\n Covid Images : {covid_images_on_data_set}\n Non Covid Images : {Total_images_on_data_set}\n")    
        
# images array shuffles
img_path_covid_old = random.sample(img_path_covid_old, len(img_path_covid_old))
img_path_non_covid_old = random.sample(img_path_non_covid_old, len(img_path_non_covid_old))

        
#  images array lenth 
divider = 10
middle_img_path_covid = int(len(img_path_covid_old)/divider)
middle_img_path_non_covid = int(len(img_path_non_covid_old)/divider)

img_path_covid =img_path_covid_old[:middle_img_path_covid]
img_path_non_covid =img_path_non_covid_old[:middle_img_path_non_covid]

total_images = len(img_path_non_covid) + len(img_path_covid)

print(f"Covid images count     : {len(img_path_covid)}")           
print(f"Non covid images count : {len(img_path_non_covid)}")   
print(f"All images count       : {total_images}") 
print(f"Covid images           : {round((len(img_path_covid)/total_images)*100, 2)} %")           
print(f"Non covid images       : {round((len(img_path_non_covid)/total_images)*100, 2)} %")  

# cantidaDatosEntrenamiento = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
# cantidaDatosPruebas = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

# # LeNet Model
# lenet_model = build_lenet_model((ancho, alto), numeroCategorias)
# lenet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# imagenes, probabilidades = cargarDatos("dataset/train/", numeroCategorias, cantidaDatosEntrenamiento, ancho, alto)
# lenet_model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60)

# # VGG16 Model
# vgg16_model = build_vgg16_model((ancho, alto, numeroCanales), numeroCategorias)
# vgg16_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# imagenes, probabilidades = cargarDatos("dataset/train/", numeroCategorias, cantidaDatosEntrenamiento, ancho, alto)
# imagenes = np.expand_dims(imagenes, axis=-1)  # Add an extra dimension for channels
# imagenes = np.repeat(imagenes, 3, axis=-1)  # Repeat the single channel to create 3 channels
# vgg16_model.fit(x=vgg_preprocess_input(imagenes), y=probabilidades, epochs=30, batch_size=60)

# # ResNet50 Model
# resnet50_model = build_resnet50_model((ancho, alto, numeroCanales), numeroCategorias)
# resnet50_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# imagenes, probabilidades = cargarDatos("dataset/train/", numeroCategorias, cantidaDatosEntrenamiento, ancho, alto)
# imagenes = np.expand_dims(imagenes, axis=-1)
# imagenes = np.repeat(imagenes, 3, axis=-1)
# resnet50_model.fit(x=resnet_preprocess_input(imagenes), y=probabilidades, epochs=30, batch_size=60)

# # Prueba de los modelos
# def evaluar_modelo(modelo, ruta_test):
#     imagenes_prueba, probabilidades_prueba = cargarDatos(ruta_test, numeroCategorias, cantidaDatosPruebas, ancho, alto)
#     resultados = modelo.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
#     print("Accuracy =", resultados[1])

# evaluar_modelo(lenet_model, "dataset/test/")
# evaluar_modelo(vgg16_model, "dataset/test/")
# evaluar_modelo(resnet50_model, "dataset/test/")

# # Guardar modelos
# lenet_model.save("models/modelo_lenet.h5")
# vgg16_model.save("models/modelo_vgg16.h5")
# resnet50_model.save("models/modelo_resnet50.h5")

# # Informe de estructura de los modelos
# lenet_model.summary()
# vgg16_model.summary()
# resnet50_model.summary()