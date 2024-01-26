import tensorflow as tf

def residual_block(x, filters, kernel_size, strides=(1, 1), activation='relu'):
    """
    A residual block with identity shortcut connection.
    """
    # Save input tensor for shortcut connection
    shortcut = x
    
    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    
    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add shortcut connection to main path
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation(activation)(x)
    
    return x


def ResNet(input_shape, num_classes):
    """
    A ResNet model for lesion segmentation.
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Resize input tensor to match ground truth mask size
    x = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(inputs)
    
    # Initial convolutional layer
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64, (3, 3), activation='relu')
    x = residual_block(x, 64, (3, 3), activation='relu')
    x = residual_block(x, 64, (3, 3), activation='relu')
    x = residual_block(x, 64, (3, 3), activation='relu')
    
    # Resize output tensor to match ground truth mask size
    x = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(x)
    
    # Output layer
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    
    return model
