from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from keras.applications.vgg19 import preprocess_input

# Cargar el modelo
modelo_cargado = load_model("models/modeloVGG19Full.h5")

# Definir el generador de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_set = test_datagen.flow_from_directory(
    'subset_test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Evaluar el modelo en el conjunto de prueba
resultados = modelo_cargado.evaluate(test_set)

# Imprimir las métricas
print("Pérdida en el conjunto de prueba:", resultados[0])
print("Precisión en el conjunto de prueba:", resultados[1])

# Obtener las predicciones para el conjunto de prueba
predicciones = modelo_cargado.predict(test_set)
etiquetas_predichas = np.argmax(predicciones, axis=1)
etiquetas_verdaderas = test_set.classes

# Calcular y mostrar la matriz de confusión
matriz_confusion = confusion_matrix(etiquetas_verdaderas, etiquetas_predichas)
print("Matriz de confusión:")
print(matriz_confusion)

# Imprimir un informe de clasificación detallado
reporte_clasificacion = classification_report(etiquetas_verdaderas, etiquetas_predichas, target_names=test_set.class_indices)
print("Reporte de clasificación:")
print(reporte_clasificacion)