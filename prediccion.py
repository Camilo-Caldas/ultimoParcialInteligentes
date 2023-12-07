from keras.models import load_model
import numpy as np
import cv2

class prediccion():
    def __init__(self,ruta,ancho,alto):
        self.modelo=load_model(ruta)
        self.alto=alto
        self.ancho=ancho

    def predecir(self,imagen):
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenes_cargadas=[]
        imagenes_cargadas.append(imagen)
        imagenes_cargadas_npa=np.array(imagenes_cargadas)
        predicciones=self.modelo.predict(x=imagenes_cargadas_npa)
        print("Predicciones=",predicciones)
        clases_mayores=np.argmax(predicciones,axis=1)
        return clases_mayores[0]
    
    # def predecir(self,imagen):
    #     imagenes_cargadas=[]
    #     imagenes_cargadas.append(imagen)
    #     imagenes_cargadas_npa=np.array(imagenes_cargadas)
    #     predicciones=self.modelo.predict(x=imagenes_cargadas_npa)
    #     print("Predicciones=",predicciones)
    #     clases_mayores=np.argmax(predicciones,axis=1)
    #     return clases_mayores[0]