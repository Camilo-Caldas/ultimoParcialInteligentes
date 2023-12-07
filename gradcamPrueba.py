import os
import cv2
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from keras.models import load_model
from gradCamClass import GradCAM  
from prediccion import  prediccion

from keras.utils import load_img, img_to_array

# Cargar el modelo desde el archivo guardado
model = load_model('models/modeloUno.h5')
#model.summary()

# image = cv2.imread('dataset/train/COVID/0.png', 0)
# image = cv2.bitwise_not(image)          # ATTENTION 
# image = cv2.resize(image, (256, 256))

# # checking how it looks 
# plt.imshow(image, cmap="gray")
# #plt.show()

# image = tf.expand_dims(image, axis=-1)     # from 84 x 84 to 84 x 84 x 1 
# image = tf.divide(image, 255)              # normalize
# #image = tf.reshape(image, [1, 256, 256, 1])  # reshape to add batch dimension

# print(image.shape) # (1, 84, 84, 1)


clases=["COVID","LUNG OPACITY","NORMAL","VIRAL PNEUMONIA"]

ancho = 256
alto = 256

miModeloCNN=prediccion("models/modeloDos.h5",ancho,alto)
imagen=cv2.imread("dataset/test/Viral Pneumonia/5000.png")

resized = cv2.resize(imagen, (224, 224))
# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = load_img("dataset/test/Viral Pneumonia/5000.png", target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

# while True:
#     cv2.imshow("imagen",imagen)
#     k=cv2.waitKey(30) & 0xff
#     if k==27:
#         break
# cv2.destroyAllWindows()

# preds = model.predict(image) 
# i = np.argmax(preds[0])
# i # 0 - great model correctly recognize, this is an odd number 
imagen = cv2.resize(imagen, (256, 256))
imagen = imagen.flatten()
#imagen = tf.reshape(imagen, (256, 256))  # Cambiar la forma a (256, 256)
#imagen = imagen[:65536]  # Asegurar que la forma sea (65536,)
print(imagen.shape)
# `conv2d_19` - remember this, we talked about it earlier 
icam = GradCAM(model, miModeloCNN, 'capa_2') 
heatmap = icam.compute_heatmap(imagen)
heatmap = cv2.resize(heatmap, (256, 256))

# image = cv2.imread('dataset/train/COVID/0.png')
# image = cv2.resize(image, (256, 256))
print(heatmap.shape, imagen.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, imagen, alpha=0.5)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(20,20)

# plt.ax[0].imshow(heatmap)
# plt.ax[1].imshow(imagen)
# plt.ax[2].imshow(output)