import tensorflow as tf
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input, preprocessing, Resizing, merging
from keras import layers, models, optimizers, losses, metrics
# from main import load_data
 
# Define small model 1
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bridge """
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    """ Model """
    model = Model(inputs, outputs)
    return model

# Define small model 2
def residual_block(x, filters, kernel_size, strides=(1, 1), activation='relu'):
    """
    A residual block with identity shortcut connection.
    """
    # Save input tensor for shortcut connection
    shortcut = x
    
    # First convolutional layer
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut connection to main path
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    
    return x


def ResNet(input_shape, num_classes):
    """
    A ResNet model for lesion segmentation.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Resize input tensor to match ground truth mask size
    x = Resizing(256, 256)(inputs)
    
    # Initial convolutional layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64, (3, 3), activation='relu')
    x = residual_block(x, 64, (3, 3), activation='relu')
    x = residual_block(x, 64, (3, 3), activation='relu')
    x = residual_block(x, 64, (3, 3), activation='relu')
    
    # Resize output tensor to match ground truth mask size
    x = Resizing(input_shape[0], input_shape[1])(x)
    
    # Output layer
    x = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    # Final Layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model


# Create ensemble of two small models
def stacked_ensemble(input_shape,num_class):
    inputs = Input(input_shape)
    
    unet_model = build_unet(input_shape)
    resnet_model = ResNet(input_shape,num_class)

    # Freeze the layers in both models to prevent them from being trained
    for layer in unet_model.layers:
        layer.trainable = False

    for layer in resnet_model.layers:
        layer.trainable = False

    # Remove the last layer of the UNet model to get the feature maps
    unet_output = unet_model.layers[-2].output

    # Remove the last two layers of the ResNet model to get the feature maps
    resnet_output = resnet_model.layers[-3].output

    # unet_output = Conv2D(1, 1, activation='sigmoid')(unet_model)
    # resnet_output = Conv2D(1, 1, activation='sigmoid')(resnet_model)

    
    concatenated_output = Concatenate()([unet_output, resnet_output])
    # concatenated_output = Concatenate()([unet_model, resnet_model])

    # Add a convolutional layer to reduce the number of channels
    # conv = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(concatenated_output)

    # Add another convolutional layer to reduce the number of channels further
    # conv = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(conv)

    # Add a final convolutional layer to produce the output
    # output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv)
    
    # unet_model = load_model('D:/segmentation/segment/files/model1.h5')
    # resnet_model = load_model('D:/segmentation/segment/files/model2.h5')
    # outputs = [model(inputs) for model in models]
    # outputs = [build_unet(inputs), ResNet(inputs,num_class)]
    # outputs = [unet_model, resnet_model]
    outputs = Conv2D(1, 1, activation='sigmoid')(concatenated_output)
    # outputs = Dense(1, activation='sigmoid')(outputs)

    y = Average()([outputs])
    
    # Define input and output tensors for each model
    # unet_input = unet_model.inputs[0]
    # unet_output = unet_model.outputs[0]

    # resnet_input = resnet_model.inputs[0]
    # resnet_output = resnet_model.outputs[0]

    # Create a new model that stacks the outputs of the U-Net and ResNet models
    # stacked_outputs = concatenate([unet_output, resnet_output])
    # ensemble_model = Model(inputs=[unet_input, resnet_input], outputs=stacked_outputs)

    ensemble_model = Model(inputs=inputs, outputs=y)
    return ensemble_model

# # Load data and split into train and test sets
# (x_train, y_train), (x_test, y_test) = load_data()
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# y_train = np.array([int(label == 5) for label in y_train])
# y_test = np.array([int(label == 5) for label in y_test])

# # Split train data into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# # Normalize data
# x_train = x_train.astype('float32') / 255.
# x_val = x_val.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# Compile and train small models
# model_1 = build_unet(input_shape = (256,256,3))
# model_1.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# history_1 = model_1.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=5, callbacks=[es, mc])


# model_2 =ResNet(input_shape=(256,256,3))
# model_2.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# history_2 = model_2.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=5, callbacks=[es, mc])

# Load best models
# loaded_models = []
# for i in range(len(models)):
#     loaded_model = load_model('best_model_{}.h5'.format(i))
#     loaded_models.append(loaded_model)

# Create stacked ensemble
# ensemble_input = load_model(models=[model_1, model_2], input_shape=(256,256,3))
# ensemble_outputs = [model(ensemble_input) for model in loaded_models]
# ensemble_output = Average()(ensemble_outputs)
# stacked_model = Model(inputs=ensemble_input, outputs=ensemble_output)

# compile and evaluate the stacked ensemble model
# stacked_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# loss, accuracy = stacked_model.evaluate(x_test, y_test)
# print('Test Loss:', loss)
# print('Test Accuracy:', accuracy)

