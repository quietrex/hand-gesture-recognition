import cv2
import numpy as np
from tensorflow import Graph, Session
import tensorflow as tf
import os; 
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

def make_inference(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.flip(image, 1)

    # Reshape
    res = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float64")
    res = res / 255
    res = np.reshape(res, (1, 128, 128, 3))

    prediction= model.predict(res)

    return prediction[0]

if __name__ == "__main__":
    import keras

    print("***************************Loaded Model for inference***************************")
    try:
        model = keras.models.load_model("cnn/models/hand_detector_15_Epoch.h5")
    except Exception as e:
        print(e)

    classes = ['garbage', 'next', 'start', 'stop']

    print('<< GARBAGE >>')
    im4 = cv2.imread("test_images\\garbage_2.png")
    print(make_inference(model, im4))
    print('Predicted: ' + classes[np.argmax(make_inference(model, im4))])
    
    print('<< NEXT >>')
    im1 = cv2.imread("test_images\\next_1.png")
    print(make_inference(model, im1))
    print('Predicted: ' + classes[np.argmax(make_inference(model, im1))])

    print('<< STOP >>')
    im2 = cv2.imread("test_images\\stop_3.png")
    print(make_inference(model, im2))    
    print('Predicted: ' + classes[np.argmax(make_inference(model, im2))])

    print('<< START >>')
    im3 = cv2.imread("test_images\\start_2.png")
    print(make_inference(model, im3))
    print('Predicted: ' + classes[np.argmax(make_inference(model, im3))])

    