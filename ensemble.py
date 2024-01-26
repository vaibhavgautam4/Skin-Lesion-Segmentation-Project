import tensorflow as tf
from keras.layers import Input
from keras.models import Model

# Define input shape
input_shape = (256, 256, 3)

# Define input layer
inputs = Input(input_shape)

# Load pre-trained UNet model
unet_model = tf.keras.models.load_model('D:/segmentation/segment/files/model1.h5')

# Freeze layers in the UNet model
for layer in unet_model.layers:
    layer.trainable = False

# Get output from UNet model
unet_output = unet_model(inputs)

# Load pre-trained ResNet model
resnet_model = tf.keras.models.load_model('D:/segmentation/segment/files/model2.h5')

# Freeze layers in the ResNet model
for layer in resnet_model.layers:
    layer.trainable = False

# Get output from ResNet model
resnet_output = resnet_model(inputs)

# Concatenate the outputs
concatenated = tf.keras.layers.concatenate([unet_output, resnet_output])

# Add some more layers if necessary
# ...

# Define the final output layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated)

# Define the model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
