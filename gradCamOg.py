import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import keras
from PIL import Image
from keras.models import load_model
from IPython.display import Image, display
import prediccion

# # Cargar el modelo desde el archivo guardado
# model = load_model('models/modeloCinco.h5', compile=False)

# # Dimensiones de entrada esperadas por el modelo
# img_size = (256, 256)

# # Nombre de la última capa convolucional
# last_conv_layer_name = "capa_5"

# def preprocess_image(img_path, size):
#     img = keras.utils.load_img(img_path, target_size=size)
#     img_array = keras.utils.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
#     last_conv_layer = model.get_layer(last_conv_layer_name)
#     grad_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if preds is None:
#             raise ValueError("Model output should not be None for Grad-CAM")

#         pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     grads = tape.gradient(class_channel, last_conv_layer_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     heatmap = tf.reduce_mean(tf.multiply(last_conv_layer_output, pooled_grads), axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()

# def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
#     img = keras.utils.load_img(img_path)
#     img = keras.utils.img_to_array(img)

#     heatmap = np.uint8(255 * heatmap)
#     jet = mpl.cm.get_cmap("viridis")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]

#     jet_heatmap = keras.utils.array_to_img(jet_heatmap)
#     jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#     jet_heatmap = keras.utils.img_to_array(jet_heatmap)

#     superimposed_img = jet_heatmap * alpha + img
#     superimposed_img = keras.utils.array_to_img(superimposed_img)

#     superimposed_img.save(cam_path)
#     display(Image(cam_path))

# # Ruta de la imagen a visualizar
# img_path = 'dataset/test/Viral Pneumonia/5000.png'

# # Preparar la imagen
# img_array = preprocess_image(img_path, size=img_size)

# # Generar el mapa de calor Grad-CAM
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# # Mostrar el mapa de calor Grad-CAM
# plt.imshow(heatmap, cmap='viridis')
# plt.show()

# # Guardar y mostrar Grad-CAM superpuesto en la imagen original
# save_and_display_gradcam(img_path, heatmap)
# Cargar el modelo desde el archivo guardado
model = load_model('models/modeloUno.h5')
clases=["COVID","LUNG OPACITY","NORMAL","VIRAL PNEUMONIA"]

ancho = 256
alto = 256

# Dimensiones de entrada esperadas por el modelo
img_size = (256, 256)

# Nombre de la última capa convolucional
last_conv_layer_name = "capa_2"


#     return img_array

def preprocess_image(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    img_array = keras.utils.img_to_array(img)
    print(f"Original shape: {img_array.shape}")  # Imprimir la forma original

    # Redimensionar la imagen a las dimensiones esperadas por la capa de entrada
    img_array = tf.image.resize(img_array, img_size)  # Ajusta el tamaño según tus necesidades

    # Calcular el número de elementos necesarios para la forma esperada
    num_elements = 65536  # Número esperado de elementos

    # Aplanar la imagen
    img_array = tf.reshape(img_array, (1, -1))

    # Calcular la diferencia entre el número de elementos actual y el esperado
    current_elements = tf.shape(img_array)[1]
    diff_elements = num_elements - current_elements

    # Rellenar la imagen con píxeles negros si es necesario
    if diff_elements > 0:
        print("entra a rellenar")
        padding = tf.zeros((1, diff_elements))
        img_array = tf.concat([img_array, padding], axis=1)
    # Recortar la imagen si es necesario
    elif diff_elements < 0:
        print("entra a recortar")
        img_array = img_array[:, :num_elements]

    print(f"Final shape: {img_array.shape}")  # Imprimir la forma final

    # Normalizar manualmente los píxeles de la imagen
    img_array /= 255.0  # Escalar los valores de píxeles al rango [0, 1]
    img_array -= 0.5    # Centrar los valores alrededor de cero
    img_array *= 2.0    # Escalar los valores para que estén en el rango [-1, 1]

    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, imagen):
    # Extraer la última capa convolucional del modelo
    #last_conv_layer = model.get_layer(last_conv_layer_name)
    # print(last_conv_layer.name)
    # print(model.inputs)
    #last_conv_model = keras.models.Model(model.inputs, last_conv_layer.output)
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Calcular las activaciones de la última capa convolucional
    #last_conv_activations = last_conv_model.predict(img_array)

    # Obtener la predicción del modelo
    #preds = last_conv_model.predict(img_array)

    #class_idx = np.argmax(preds[0])
    #print('preds size',len(preds))
    imagen = cv2.imread(imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.resize(imagen, (256, 256))
    imagen = imagen.flatten()
    imagen = imagen / 255
    imagenes_cargadas=[]
    imagenes_cargadas.append(imagen)
    imagenes_cargadas_npa=np.array(imagenes_cargadas)
    predicciones=model.predict(x=imagenes_cargadas_npa)
    print("Predicciones=",predicciones)
    clases_mayores=np.argmax(predicciones,axis=1)
    class_idx = None
    print("La imagen cargada es ",clases[clases_mayores[0]])
    

    # # Then, we compute the gradient of the top predicted class for our input image
    # # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        #last_conv_layer_output = last_conv_model(img_array)
        last_conv_layer_output, preds = grad_model(img_array)
        
        class_idx = tf.argmax(preds[0])

        print('class index ',class_idx)
        class_channel = preds[:, class_idx]
        print('class channel ',class_channel)

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



    # # Calcular el gradiente respecto a la clase predicha
    # grads = model.optimizer.get_gradients(model.output[:, class_idx], last_conv_layer.output)[0]
    # pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # heatmap = tf.reduce_mean(tf.multiply(last_conv_activations, pooled_grads), axis=-1)

    # # Normalizar el mapa de calor entre 0 y 1
    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)

    # return heatmap



def save_and_display_gradcam(img_path, heatmap, cam_path="Neumonia2.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    cam_path = "./mapas_calor/"+cam_path
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

# Ruta de la imagen a visualizar
img_path = 'dataset/test/COVID/5000.png'

# Preparar la imagen
img_array = preprocess_image(img_path, size=img_size)

# Generar el mapa de calor Grad-CAM
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, img_path)

# Mostrar el mapa de calor Grad-CAM
plt.imshow(heatmap, cmap='viridis')
plt.show()

# Guardar y mostrar Grad-CAM superpuesto en la imagen original
save_and_display_gradcam(img_path, heatmap)