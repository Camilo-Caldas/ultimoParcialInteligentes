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

image = cv2.imread('/dataset/train/COVID/0.png', 0)
image = cv2.bitwise_not(image)          # ATTENTION 
image = cv2.resize(image, (84, 84))

# checking how it looks 
plt.imshow(image, cmap="gray")
plt.show()

image = tf.expand_dims(image, axis=-1)     # from 84 x 84 to 84 x 84 x 1 
image = tf.divide(image, 255)              # normalize
image = tf.reshape(image, [1, 84, 84, 1])  # reshape to add batch dimension

print(image.shape) # (1, 84, 84, 1)

preds = model.predict(image) 
i = np.argmax(preds[0])
i # 0 - great model correctly recognize, this is an odd number 

# `conv2d_19` - remember this, we talked about it earlier 
icam = GradCAM(model, i, 'conv2d_19') 
heatmap = icam.compute_heatmap(image)
heatmap = cv2.resize(heatmap, (84, 84))

image = cv2.imread('/content/5.png')
image = cv2.resize(image, (84, 84))
print(heatmap.shape, image.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(20,20)

ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)