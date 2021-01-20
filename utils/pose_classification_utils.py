import cv2
import numpy as np
from tensorflow import Graph, Session
import tensorflow as tf
import os; 
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

def load_KerasGraph(path): 
    print("> ====== loading Keras model for classification")
    thread_graph = Graph()
    with thread_graph.as_default():
        thread_session = Session()
        with thread_session.as_default():
            model = keras.models.load_model(path)
            #model._make_predict_function()
            graph = tf.get_default_graph()
    print(">  ====== Keras model loaded")
    return model, graph, thread_session

def classify(model, graph, sess, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = cv2.flip(im, 1)

    # Reshape
    res = cv2.resize(im, (128,128), interpolation=cv2.INTER_AREA)

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float64")
    res = res / 255
    res = np.reshape(res, (1, 128, 128, 3))

    with graph.as_default():
        with sess.as_default():
            prediction= model.predict(res)

    return prediction[0] 
