import cv2
import numpy as np
from random import randint
CLASSES = ['garbage', 'next', 'start', 'stop']

def drawInferences(values, names=['', '', '', '', '', '']):
    print(values)
    print("Predicted: " + CLASSES[np.argmax(values)])
    # cv2.imshow("Inferences", blank)

