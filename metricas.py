
from keras.models import load_model
import cargaData
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

ancho = 256
alto = 256
pixeles = ancho * alto
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
nombreCategorias = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

cantidaDatosEntrenamiento = [6996, 6571, 6571, 4292]
cantidaDatosPruebas = [8745, 7674, 8214, 5365]
imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas("dataset/test/", nombreCategorias,
                                                                      cantidaDatosPruebas, cantidaDatosEntrenamiento, ancho, alto)
# Cargar el modelo desde el archivo guardado
loaded_model = load_model('models/modeloDos.h5')

# Predecir en datos de prueba
loaded_y_pred = loaded_model.predict(imagenesPrueba)
loaded_y_pred_classes = np.argmax(loaded_y_pred, axis=1)
loaded_y_true_classes = np.argmax(probabilidadesPrueba, axis=1)

# Calcular métricas por separado
accuracy = accuracy_score(loaded_y_true_classes, loaded_y_pred_classes)
precision = precision_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')
recall = recall_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')
f1 = f1_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Obtener la matriz de confusión
conf_mat = confusion_matrix(loaded_y_true_classes, loaded_y_pred_classes)

print(conf_mat)

# Mostrar la matriz de confusión como un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=nombreCategorias, yticklabels=nombreCategorias)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()