import os
import random
import shutil
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

# Define the paths
subset_train_path = "subset_train"
subset_test_path = "subset_test"

# Function to create a subset of the data
def create_subset(original_path, subset_path, subset_fraction):
    classes = os.listdir(original_path)
    os.makedirs(subset_path, exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(original_path, class_name)
        subset_class_path = os.path.join(subset_path, class_name)
        os.makedirs(subset_class_path, exist_ok=True)

        images = os.listdir(class_path)
        random.shuffle(images)
        subset_size = int(subset_fraction * len(images))
        selected_images = images[:subset_size]

        for image in selected_images:
            original_image_path = os.path.join(class_path, image)
            subset_image_path = os.path.join(subset_class_path, image)
            shutil.copyfile(original_image_path, subset_image_path)

# Create a subset of the training data
create_subset(train_path, subset_train_path, subset_fraction=0.8)

# Create a subset of the test data
create_subset(test_path, subset_test_path, subset_fraction=0.2)

# Use flow_from_directory with the subset
train_set = train_datagen.flow_from_directory(subset_train_path,
                                              target_size=(256, 256),
                                              batch_size=300,
                                              class_mode='categorical')

test_set = test_datagen.flow_from_directory(subset_test_path,
                                            target_size=(256, 256),
                                            batch_size=300,
                                            class_mode='categorical')



model.fit( train_set,
                          validation_data = test_set,
                          epochs = 30,
                          steps_per_epoch = 25,
                          validation_steps = 3,
                          verbose = 2)

ruta = "models/modeloVGG19Full.h5"
model.save(ruta)
model.summary()