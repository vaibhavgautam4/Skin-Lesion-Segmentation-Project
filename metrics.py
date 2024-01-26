import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
import matplotlib as plt
from sklearn.metrics import roc_auc_score, roc_curve


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def sensitivity(y_true, y_pred):
    def f(y_true, y_pred):
        true_positives = (y_pred * y_true).sum()
        false_negatives = ((1 - y_pred) * y_true).sum()
        x = true_positives / (true_positives + false_negatives)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

import tensorflow as tf

# def cal_sensitivity(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=tf.float32))
#     false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), dtype=tf.float32))
#     sensitivity = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
#     return sensitivity

def specificity(y_true, y_pred):
    def f(y_true, y_pred):
        true_negatives = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0), dtype=tf.float32))
        actual_negatives = K.sum(tf.cast(K.equal(y_true, 0), dtype=tf.float32))
        x = true_negatives / (actual_negatives + K.epsilon())
        # x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def cal_specificity(y_pred, y_true):
    def f(y_true, y_pred):
        TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
        FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
        x = TN / (TN + FP + tf.keras.backend.epsilon())
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

 # roc curve
    
# fpr, tpr, thresholds = roc_curve(y, y_pred)
# roc_auc = roc_auc_score(fpr, tpr)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")
# plt.show()

def RocCurve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()