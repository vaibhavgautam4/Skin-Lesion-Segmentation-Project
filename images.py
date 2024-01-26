import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
# import pandas as pd
from glob import glob

import tensorflow as tf
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from unet import build_unet
from metrics import dice_coef, iou, sensitivity, specificity, cal_specificity
# from model_resunet import build_resunet
# from resnet import build_resnet
from resnet import ResNet
# from stack_ensemble import stacked_ensemble

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.metrics import Recall, Precision
from keras.layers import Dense, Dropout

h =256
w =256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_dir_paths(path1,path2,path3):
    path = os.path.join(path1,path2,path3)
    if not os.path.exists(path):
        os.makedirs(path)        

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path, split = 0.2):
    training = sorted(glob(os.path.join(dataset_path, "Train/*", "*.jpg")))
    # training = sorted(glob(os.path.join(dataset_path, "HAM10000_images", "*.jpg")))
    testing = sorted(glob(os.path.join(dataset_path, "Valid/*", "*.jpg")))
    # testing = sorted(glob(os.path.join(dataset_path, "HAM10000_images", "*.jpg")))
    test_size = int(len(training) * split)
    # test_size = 118
    print(len(training))
    print(len(testing))

    x_train , x_valid = train_test_split(training , test_size = test_size , random_state=42)
    y_train , y_valid = train_test_split(testing , test_size= test_size , random_state=42)

    x_train , x_test = train_test_split(training , test_size = test_size , random_state=42)
    y_train , y_test = train_test_split(testing , test_size= test_size , random_state=42)
    return (x_train,y_train), (x_valid,y_valid), (x_test,y_test)
    # return training, testing

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR) ##(h,w,3)
    # Convert the image to grayscale
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # Define the kernel size for the median filter
    kernel_size = 5
    # Apply the median filter to the image
    x = cv2.medianBlur(x, kernel_size)
    x = cv2.resize(x , (w ,h))
    x = x/255.0
    x = x.astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x                                ##(256,256,3)

def read_masks(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ##(h,w)
    # Convert the image to grayscale
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # Define the kernel size for the median filter
    kernel_size = 5
    # Apply the median filter to the image
    x = cv2.medianBlur(x, kernel_size)
    x = cv2.resize(x , (w ,h))
    x = x/255.0
    x = x.astype(np.float32)                    ##(256,256)
    x = np.expand_dims(x, axis = -1)
    return x                                    

def tf_parse(x,y):
    def _parse(x, y):
        x = read_image(x)
        y = read_masks(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([h, w, 3])
    y.set_shape([h, w, 1])
    return x, y

def tf_dataset(x, y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":
    """seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
# 
    # create_dir("files")

    """hyperparameters"""
    batch_size = 32
    # lr = 1e-4 ##(0.0001)
    # num_epoch = 10
    # model_path = "model/model1.h5"
    # csv_path = "model/data1.csv"
    
    # model_path = "model/model2.h5"
    # csv_path = "model/data2.csv"
    
    # model_path = "model/model.h5"
    # csv_path = "model/data.csv"

    """dataset : 60/20/20"""
    dataset_path = "D:/segmentation/segment/usic/Skin cancer ISIC The International Skin Imaging Collaboration"
    # dataset_path = "D:/segmentation/segment/ham10000"
    (x_train,y_train) , (x_valid,y_valid), (x_test,y_test) = load_data(dataset_path, 0.2)
    # x_train , x_test= load_data(dataset_path, 0.2)

    print(f"Train : {len(x_train)} - {len(y_train)}")
    print(f"masks : {len(x_valid)} - {len(y_valid)}")
    print(f"test : {len(x_test)} - {len(y_test)}")

    train_dataset = tf_dataset(x_train, y_train, batch = batch_size)
    valid_dataset = tf_dataset(x_valid, y_valid, batch = batch_size)

    train_step = len(x_train)//batch_size
    valid_step = len(x_valid)//batch_size

    if len(x_train) % batch_size != 0:
        train_step += 1

    if len(x_valid) % batch_size !=0:
        valid_step += 1

    print(train_step)
    print(valid_step)
    print(read_image)
    """model"""

    # model = build_unet([h,w,3])
    # model = build_resunet([h,w,3])
    # model =  build_resnet()
    # model = ResNet((h,w,3), 1)

    # model = stacked_ensemble((h,w,3), 1)

    # Create UNet model
    # unet_model = build_unet(input_shape=(h, w, 3))
    # model = build_unet(input_shape=(h, w, 3))

    # Create ResNet model
    # resnet_model = ResNet(input_shape=(h, w, 3), num_classes=1)

    # Create input tensor
    # inputs = tf.keras.layers.Input(shape=(h, w, 3))

    # Resize input tensor for ResNet model
    # resized_inputs = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(inputs)


    # Create UNet and ResNet output tensors
    # unet_output = unet_model(inputs)
    # resnet_output = resnet_model(resized_inputs)

    # Resize ResNet output tensor
    # resized_unet_output = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(unet_output)
    # resized_resnet_output = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(resnet_output)


    # Concatenate output tensors
    # concatenated_output = tf.keras.layers.Concatenate()([resized_unet_output, resized_resnet_output])

    # Add dense layers for stacking ensemble
    # dense = Dense(256, activation='relu')(concatenated_output)
    # dropout = Dropout(0.5)(dense)
    # output = Dense(1, activation='sigmoid')(dropout)

    # Create stacked ensemble model
    # model = tf.keras.models.Model(inputs=inputs, outputs=output)

    # Compile stacked ensemble model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # print("model built!")
    # metric = [iou, dice_coef, sensitivity, specificity, Recall(), Precision()]
    # print("metrics built!")


    # model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics= metric)
    # ensemble_model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics= metric)
    # print("compiled!!")
    # model.summary()
    # ensemble_model.summary()

    # callback = [
    #     ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    #     ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    #     CSVLogger(csv_path),
    #     TensorBoard(),
    #     EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=False)
    # ]

    # model.fit(
    #     train_dataset,
    #     epochs = num_epoch,
    #     batch_size = batch_size, 
    #     validation_data = valid_dataset,
    #     steps_per_epoch = train_step,
    #     validation_steps = valid_step,
    #     callbacks = callback
    # )
