import os

carpeta = 'dataset/train/Viral Pneumonia'  # Reemplaza con la ruta de tu carpeta

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta)

# Filtrar solo los archivos (no directorios)
archivos = [archivo for archivo in archivos if os.path.isfile(os.path.join(carpeta, archivo))]

# Ordenar los archivos por nombre
archivos.sort()

# Renombrar los archivos con numeración secuencial
for i, archivo in enumerate(archivos, start=0):
    nuevo_nombre = f"{i}.png"  # Puedes ajustar el formato según tus necesidades
    ruta_antigua = os.path.join(carpeta, archivo)
    ruta_nueva = os.path.join(carpeta, nuevo_nombre)
    os.rename(ruta_antigua, ruta_nueva)

print("¡Proceso completado!")