import tensorflow as tf
import matplotlib.pyplot as plt
import numpy  as np
import os
import datetime
import pandas as pd
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def tf_confusion_metrics(predict, real):
 
    predictions=np.array(predict)
    actuals=np.array(real)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    predictions = tf.where(predictions > 0.25, ones_like_predictions, zeros_like_predictions)
    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )
 
    tn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
    )
 
    fp_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ),
          "float"
        )
    )
 
    fn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
    )
    #tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op],feed_dict={predict: predictLabel , real:test_y})
    tp=np.array(tp_op)
    tn=np.array(tn_op)
    fp=np.array(fp_op)
    fn=np.array(fn_op)
 
    tpr = float(tp)/(float(tp) + float(fn))
    fpr = float(fp)/(float(fp) + float(tn))
    fnr = float(fn)/(float(tp) + float(fn))
 
    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp)/(float(tp) + float(fp)+0.0001)
 
    f1_score = (2 * (precision * recall)) / (precision + recall+0.0001)
    return [accuracy,precision,recall,f1_score]

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, padding='same'):
    # the first layer
    X_shortcut=input_tensor
    x = tf.keras.layers.Conv2D(n_filters/4, 1, padding=padding,dilation_rate=1)(
        input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #x = tf.keras.layers.Dropout(rate=0.1)(x)
    # the second layer
    x = tf.keras.layers.Conv2D(n_filters/4, kernel_size, padding=padding,dilation_rate=2)(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #X = tf.keras.layers.Dropout(rate=0.1)(X)
    x = tf.keras.layers.Conv2D(n_filters , 1, padding=padding,dilation_rate=2)(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x=tf.add(x,input_tensor)
    X = tf.keras.layers.Activation('relu')(x)
    return X

def conv2d_block_3(input_tensor, n_filters, kernel_size=3, batchnorm=True, padding='same'):
    # the first layer
    X_shortcut=tf.keras.layers.Conv2D(n_filters, kernel_size, padding=padding,dilation_rate=1)(
        input_tensor)
    if batchnorm:
        X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)
    #X_shortcut = tf.keras.layers.Activation('relu')(X_shortcut)
    
    x = tf.keras.layers.Conv2D(n_filters/4, 1, padding=padding,dilation_rate=1)(
        input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #x = tf.keras.layers.Dropout(rate=0.1)(x)
    # the second layer
    x = tf.keras.layers.Conv2D(n_filters/4, kernel_size, padding=padding,dilation_rate=2)(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #X = tf.keras.layers.Dropout(rate=0.1)(X)
    x = tf.keras.layers.Conv2D(n_filters , 1, padding=padding,dilation_rate=2)(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x=tf.add(x,X_shortcut)
    X = tf.keras.layers.Activation('relu')(x)
    return X

def conv2d_block_4(input_tensor, n_filters, kernel_size=3, batchnorm=True, padding='same'):
    # the first layer
    
    X_shortcut=tf.keras.layers.Conv2D(n_filters, kernel_size=2, strides=2, padding='valid',dilation_rate=1)(
        input_tensor)
    if batchnorm:
        X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)
    #X_shortcut = tf.keras.layers.Activation('relu')(X_shortcut)
    
    x = tf.keras.layers.Conv2D(n_filters/4, kernel_size=2, strides=2,padding='valid',dilation_rate=1)(
        input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #x = tf.keras.layers.Dropout(rate=0.1)(x)
    # the second layer
    x = tf.keras.layers.Conv2D(n_filters/4, kernel_size, padding=padding,dilation_rate=2)(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #X = tf.keras.layers.Dropout(rate=0.1)(X)
    x = tf.keras.layers.Conv2D(n_filters , 1, padding=padding,dilation_rate=2)(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x=tf.add(x,X_shortcut)
    X = tf.keras.layers.Activation('relu')(x)
    return X

def conv2d_block_2(input_tensor, n_filters, kernel_size=3, batchnorm=True, padding='same'):
    x = tf.keras.layers.Conv2D(n_filters, kernel_size, padding=padding,dilation_rate=1)(
        input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    X = tf.keras.layers.Activation('relu')(x)
    return X

if __name__ == "__main__":
    keras_model_path = 'keres_model_1'
    restored_keras_model = tf.keras.models.load_model(keras_model_path,custom_objects={'conv2d_block': conv2d_block,'conv2d_block_2': conv2d_block_2,'conv2d_block_3': conv2d_block_3,'conv2d_block_4': conv2d_block_4},compile=False)
    
    data=np.load(r'data1.npy')
    p=np.max(np.abs(data))
    x_test=data.reshape(1,64,64)/p
    
    #draw data
    plt.figure(figsize=(6,5))
    ct1=plt.contourf(np.array(x_test).reshape(64,64))
    plt.axis('equal')
    cbar = plt.colorbar(ct1)
    
    #predict edge
    pred_anno = restored_keras_model.predict(x_test)
    pred_anno_1=np.array(pred_anno).reshape(1,4096)
    
    #draw result
    plt.figure(figsize=(6,5))
    fig=plt.pcolor(pred_anno_1.reshape(64,64))
    plt.axis('equal')
    cbar1 = plt.colorbar(fig)
    
    