from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from keras.applications.vgg19 import preprocess_input
import cargaData

ancho = 256
alto = 256
pixeles = ancho * alto
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
nombreCategorias = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

cantidaDatosEntrenamiento = [6996, 6571, 6571, 4292]
cantidaDatosPruebas = [8745, 7674, 8214, 5365]
# imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas_vgg("dataset/test/", nombreCategorias,
#                                                                       cantidaDatosPruebas, cantidaDatosEntrenamiento, ancho, alto)

# Cargar el modelo
modelo_cargado = load_model("models/modeloVGG19.h5")

# Definir el generador de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_set = test_datagen.flow_from_directory(
    'subset_test',
    target_size=(256, 256),
    batch_size=30,
    class_mode='categorical'
)

# Evaluar el modelo en el conjunto de prueba
resultados = modelo_cargado.evaluate(test_set)

# Imprimir las métricas
print("Métricas del modelo cargado: ", resultados)
print("Pérdida en el conjunto de prueba:", resultados[0])
print("Precisión en el conjunto de prueba:", resultados[1])

# Obtener las predicciones para el conjunto de prueba
#predicciones = modelo_cargado.predict(test_set)
# etiquetas_predichas = np.argmax(predicciones, axis=1)
# etiquetas_verdaderas = test_set.classes

# # Calcular y mostrar la matriz de confusión
# matriz_confusion = confusion_matrix(etiquetas_verdaderas, etiquetas_predichas)
# print("Matriz de confusión:")
# print(matriz_confusion)

# # Imprimir un informe de clasificación detallado
# reporte_clasificacion = classification_report(etiquetas_verdaderas, etiquetas_predichas, target_names=test_set.class_indices)
# print("Reporte de clasificación:")
# print(reporte_clasificacion)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np

# Obtener las predicciones para el conjunto de prueba
predicciones = modelo_cargado.predict(test_set)
print(predicciones)
etiquetas_predichas = np.argmax(predicciones, axis=1)
etiquetas_verdaderas = test_set.classes

# Calcular métricas
accuracy = accuracy_score(etiquetas_verdaderas, etiquetas_predichas)
precision = precision_score(etiquetas_verdaderas, etiquetas_predichas, average='weighted')
recall = recall_score(etiquetas_verdaderas, etiquetas_predichas, average='weighted')
f1 = f1_score(etiquetas_verdaderas, etiquetas_predichas, average='weighted')

# Imprimir métricas
print("Exactitud en el conjunto de prueba:", accuracy)
print("Precisión en el conjunto de prueba:", precision)
print("Recall en el conjunto de prueba:", recall)
print("F1-score en el conjunto de prueba:", f1)

import seaborn as sns
import matplotlib.pyplot as plt

# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(etiquetas_verdaderas, etiquetas_predichas)

# Mostrar la matriz de confusión gráficamente
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=test_set.class_indices.keys(), yticklabels=test_set.class_indices.keys())
plt.title('Matriz de Confusión')
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Verdaderas')
plt.show()