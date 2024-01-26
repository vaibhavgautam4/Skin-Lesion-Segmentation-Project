import numpy as np
import tensorflow as tf
import cv2
import os

h = 256
w =256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR) ##(h,w,3)
    # resize the image
    x = cv2.resize(x , (h ,w))

    ori_x = x
    # ori_x = cv2.resize(ori_x, (h, w))
    
    # Convert the image to grayscale
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    
    # Define the kernel size for the median filter
    # kernel_size = 5
    # Apply the median filter to the image
    # x = cv2.medianBlur(x, kernel_size)


    # normalize the image
    x = x/255.0
    x = x.astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    # return the images
    return ori_x, x                                ##(256,256,3)

def save_results(ori_x, x, save_image_path):
    line = np.ones((h, 10, 3)) * 255
    print(ori_x.shape)
    print(x.shape)
    
    cat_images = np.concatenate([ori_x, line, x * 255], axis = 1)
    cv2.imwrite(save_image_path, cat_images)


if __name__ == '__main__':
    save_path = "D:/segmentation/segment/files/pre process/ISIC_0025780.jpg"
    pre = "D:/segmentation/segment/files/pre process"
    path = "D:/segmentation/segment/usic/Skin cancer ISIC The International Skin Imaging Collaboration/Train/actinic keratosis/ISIC_0025780.jpg"

    ori_x , x = read_image(path)
    save_results(ori_x, x, save_path)
    print(len(os.listdir(pre)))





    