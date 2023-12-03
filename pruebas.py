import cv2
from prediccion import  prediccion

clases=["COVID","LUNG OPACITY","NORMAL","VIRAL PNEUMONIA"]

ancho = 256
alto = 256

miModeloCNN=prediccion("models/modeloDos.h5",ancho,alto)
imagen=cv2.imread("dataset/test/Viral Pneumonia/5000.png")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()