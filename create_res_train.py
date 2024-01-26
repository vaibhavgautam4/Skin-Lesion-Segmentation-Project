import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from main import load_data, create_dir, create_dir_paths

h=256
w=256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR) ##(h,w,3)
    x = cv2.resize(x , (w ,h))
    ori_x = x                               ##(h,w)
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis =0)
    return ori_x, x                                ##(1, 256,256,3)

def read_masks(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ##(h,w)
    x = cv2.resize(x , (w ,h))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)                    ##(256,256)
    # x = np.expand_dims(x, axis = -1)
    return ori_x, x 

def save_results(ori_x, y_pred, save_image_path):
    line = np.ones((h, 10, 3)) * 255

    # ori_y = np.expand_dims(ori_y, axis =-1)
    # ori_y = np.concatenate([ori_y,ori_y,ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis =-1)
    print(y_pred.shape)
    # y_pred = np.squeeze(y_pred , axis=2)
    # print(y_pred.shape)
    y_pred = np.concatenate([y_pred,y_pred,y_pred], axis=-1)
    y_pred = y_pred*255
    print(y_pred.shape)
    # print(ori_x.shape)
    
    cat_images = np.concatenate([ori_x, line, y_pred * 255, line], axis = 1)
    # window = 'image'
    # cv2.imshow(window, cat_images)
    cv2.imwrite(save_image_path, cat_images )

def save_val(ori_x, y_pred, save_image_path):
    # line = np.ones((h, 10, 3)) * 255

    # ori_y = np.expand_dims(ori_y, axis =-1)
    # ori_y = np.concatenate([ori_y,ori_y,ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis =-1)
    print(y_pred.shape)
    # y_pred = np.squeeze(y_pred , axis=2)
    # print(y_pred.shape)
    y_pred = np.concatenate([y_pred,y_pred,y_pred], axis=-1)
    y_pred = y_pred*255
    print(y_pred.shape)
    # print(ori_x.shape)
    
    # cat_images = np.concatenate([ori_x, line, y_pred * 255, line], axis = 1)
    # window = 'image'
    # cv2.imshow(window, cat_images)
    cv2.imwrite(save_image_path, y_pred )

if __name__ == "__main__":
    """seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    results = create_dir("result train")
    """load model"""
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model('files/model/model2.h5')
        model.summary()

"""load test data"""
dataset_path = "D:/segmentation/segment/usic/Skin cancer ISIC The International Skin Imaging Collaboration"
(x_train, y_train), (x_valid, y_valid), (x_test, y_test)  = load_data(dataset_path)
# x_train, x_test  = load_data(dataset_path)

SCORE = []
for x, y in tqdm(zip(x_train, y_train), total= len(x_train)):
    """extracting the image name"""
    print(x)
    folder = x.split("\\")[-2]
    print(folder)
    name = x.split("\\")[-1]
    print(name)
    # name = name.split(".")[0]
    # print(name)
    """read the image and mask"""
    ori_x, x =read_image(x)
    ori_y, y = read_masks(y)

    """predict the mask"""
    y_pred = model.predict(x)[0] > 0.5 
    # print(y_pred.shape)
    # print(ori_x.shape)
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)
    # print(y_pred.shape)

    """saving the predicted mask"""

    
    # mkdir()
    # create_dir_paths("usic", "Skin cancer ISIC The International Skin Imaging Collaboration" ,"valid")
    save_image_path = f"result train/{folder}/{name}"
    # val_image_path = f"D:/segmentation/segment/usic/Skin cancer ISIC The International Skin Imaging Collaboration/valid/{folder}/{name}"
    # save_image_path1 = f"{save_image_path}/{name}"
    save_results(ori_x, y_pred, save_image_path)
    # save_val(ori_x, y_pred, val_image_path)
    print(len(os.listdir("results")))

#     """ Flatten the array """
#     y = y.flatten()
#     y_pred = y_pred.flatten()

#     """ Calculating metrics values """
#     acc_value = accuracy_score(y, y_pred)
#     f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
#     jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
#     recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
#     precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
#     SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

# """ mean metrics values """
# score = [s[1:] for s in SCORE]
# score = np.mean(score, axis=0)
# print(f"Accuracy: {score[0]:0.5f}")
# print(f"F1: {score[1]:0.5f}")
# print(f"Jaccard: {score[2]:0.5f}")
# print(f"Recall: {score[3]:0.5f}")
# print(f"Precision: {score[4]:0.5f}")

# df = pd.DataFrame(SCORE, columns = ["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
# df.to_csv("files/score.csv")