import datetime
import os
import random
from typing import Concatenate
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, MaxPool2D, Flatten, Conv2D, Dense, Reshape, Dropout, Activation, Concatenate
from keras.applications import VGG16, ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from keras.callbacks import ModelCheckpoint

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

print(f" Total Images on the data set : {Total_images_on_data_set}\n Covid Images : {covid_images_on_data_set}\n Non Covid Images : {covid_non_images_on_data_set}\n")    
        
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

start_time = datetime.datetime.now()
print(f"Start images reading :{start_time}")
img_size=256
data=[]
target=[]

def image_add(img_path,category) :
    img=cv2.imread(img_path)
    try:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
                    #Coverting the image into gray scale
        resized=cv2.resize(gray,(img_size,img_size))
                    #resizing the gray scale into 256x256, since we need a fixed common size for all the images in the dataset
        data.append(resized)
        target.append(label_dict[category])
                    #appending the image and the label(categorized) into the list (dataset)

    except Exception as e:
        print('Exception:',e)
                    #if any exception rasied, the exception will be printed here. And pass to the next image   

for category in categories:
    if(category == "COVID-19"):
        for img_path in img_path_covid:
            image_add(img_path,category)  
    else:
        for img_path in img_path_non_covid:
            image_add(img_path,category)  

                    
end_time = datetime.datetime.now()                 
print(f"End images reading :{end_time}")

print(f"Total time taken {end_time-start_time}")

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)


new_target=keras.utils.to_categorical(target)


print(f"Data shape : {data.shape} \nTarget shape : {target.shape} \nNew Target shape : {new_target.shape}")

input_shape=data.shape[1:] #50,50,1
inp=Input(shape=input_shape)
convs=[]

parrallel_kernels=[3,5,7,9]

for k in range(len(parrallel_kernels)):
    if(k !=0):
        conv = Conv2D(128, kernel_size = k,padding = 'same' ,activation='relu')(inp)

        convs.append(conv)

out = Concatenate()(convs)
conv_model = Model(inp, out)

model = Sequential()
model.add(conv_model)

# Hidden Layer 1
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Hidden Layer 2
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Hidden Layer 3
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Output layer
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4,input_dim=128,activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.3)

print(type (train_data))
print(f"Training Data shape : {train_data.shape} \nTraining Target shape : {train_target.shape}")
print(f"Test Data shape : {test_data.shape} \nTest Target shape : {test_target.shape}")

# start_time = datetime.datetime.now()
# print(f"Start checkpoint creation:{start_time}")

# checkpoint = ModelCheckpoint('model-{epoch:03d}.model',
#                              monitor='val_accuracy',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='max')


# history=model.fit(train_data,
#                   train_target,
#                   epochs=25,
#                   validation_split=0.1)
#end_time = datetime.datetime.now()


# print(f"End checkpoint creation:{end_time}")
# print(f"Total time taken {end_time-start_time}")

# floting values in to graph

# sns.set()
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(0,len(acc))
# fig = plt.gcf()
# fig.set_size_inches(16, 8)

# plt.plot(epochs, acc, 'r', label='Training accuracy',marker = "o")
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy',marker = "o")
# plt.title('Training and validation accuracy Model 1')
# plt.xticks(np.arange(0, len(acc), 10))
# plt.legend(loc=0)
# plt.figure()

# fig = plt.gcf()
# fig.set_size_inches(16, 8)
# plt.plot(epochs, loss, 'r', label='Training Loss',marker = "o")
# plt.plot(epochs, val_loss, 'b', label='Validation Loss',marker = "o")
# plt.title('Training and validation Loss Model 1')
# plt.xticks(np.arange(0, len(acc), 10))
# plt.legend(loc=0)
# #plt.savefig('Multiclass Model .png')
# plt.figure()
# plt.show()

print(model.evaluate(test_data,test_target))

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