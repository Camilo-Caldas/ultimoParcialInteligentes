from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
# class GradCAM:
# 	def __init__(self, model, classIdx, layerName=None):
# 		# store the model, the class index used to measure the class
# 		# activation map, and the layer to be used when visualizing
# 		# the class activation map
# 		self.model = model
# 		self.classIdx = classIdx
# 		self.layerName = layerName
# 		# if the layer name is None, attempt to automatically find
# 		# the target output layer
# 		if self.layerName is None:
# 			self.layerName = self.find_target_layer()
			
# 	def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
# 		# apply the supplied color map to the heatmap and then
# 		# overlay the heatmap on the input image
# 		heatmap = cv2.applyColorMap(heatmap, colormap)
# 		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
# 		# return a 2-tuple of the color mapped heatmap and the output,
# 		# overlaid image
# 		return (heatmap, output)		
	
# 	def compute_heatmap(self, image, eps=1e-8):
# 		# construct our gradient model by supplying (1) the inputs
# 		# to our pre-trained model, (2) the output of the (presumably)
# 		# final 4D layer in the network, and (3) the output of the
# 		# softmax activations from the model
# 		gradModel = Model(
# 			inputs=[self.model.inputs],
# 			outputs=[self.model.get_layer(self.layerName).output,
# 				self.model.output])
#         # record operations for automatic differentiation
# 		with tf.GradientTape() as tape:
# 			# cast the image tensor to a float-32 data type, pass the
# 			# image through the gradient model, and grab the loss
# 			# associated with the specific class index
# 			inputs = tf.cast(image, tf.float32)
# 			# flattened_inputs = tf.reshape(inputs, (1, -1))			
# 			(convOutputs, predictions) = gradModel(inputs)
# 			loss = predictions[:, self.classIdx]
# 		# use automatic differentiation to compute the gradients
# 		grads = tape.gradient(loss, convOutputs)

# 		castConvOutputs = tf.cast(convOutputs > 0, "float32")
# 		castGrads = tf.cast(grads > 0, "float32")
# 		guidedGrads = castConvOutputs * castGrads * grads
# 		# the convolution and guided gradients have a batch dimension
# 		# (which we don't need) so let's grab the volume itself and
# 		# discard the batch
# 		convOutputs = convOutputs[0]
# 		guidedGrads = guidedGrads[0]

# 		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
# 		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

# 		(w, h) = (image.shape[2], image.shape[1])
# 		heatmap = cv2.resize(cam.numpy(), (w, h))
# 		# normalize the heatmap such that all values lie in the range
# 		# [0, 1], scale the resulting values to the range [0, 255],
# 		# and then convert to an unsigned 8-bit integer
# 		numer = heatmap - np.min(heatmap)
# 		denom = (heatmap.max() - heatmap.min()) + eps
# 		heatmap = numer / denom
# 		heatmap = (heatmap * 255).astype("uint8")
# 		# return the resulting heatmap to the calling function
# 		return heatmap
	
def grad_cam(model, img,
             layer_name="block5_conv3", label_name=None,
             category_id=None):
    """Get a heatmap by Grad-CAM.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id is None:
            category_id = np.argmax(predictions[0])
        if label_name is not None:
            print(label_name[category_id])
        output = predictions[:, category_id]
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return np.squeeze(heatmap)