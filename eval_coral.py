import cv2
from PIL import Image
import argparse
import numpy as np
from skimage import transform
import time
import tensorflow.lite as tflite
#import tflite_runtime.interpreter as tflite

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

mean = np.array([0.6661, 0.6211, 0.5492])
std = np.array([0.2871, 0.2917, 0.3310])

test_data = np.load('test_data.npz')
images = test_data['arr_0']

test_label = np.load('test_label.npz')
labels = test_label['arr_0']

model = tflite.Interpreter(model_path='model_classify_no_coral.tflite')
#model = tflite.Interpreter(model_path='model_classify_edgetpu.tflite',experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

model.allocate_tensors()

out_data = []
input_details = model.get_input_details()
output_details = model.get_output_details()
num_examples = 0
num_correct = 0
for idx, image in enumerate(images):
    image = cv2.resize(image, (150, 150))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.divide(image, 255)
    image = np.subtract(image, mean)
    image = np.divide(image, std)
    image = image.astype(np.float32)
    model.set_tensor(input_details[0]['index'], np.expand_dims(image, 0))
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])
    softmaxed_output = softmax(predictions)
    predictions = np.argmax(softmaxed_output)
    num_correct += int(np.equal(predictions, labels[idx]))
    num_examples += 1

print('Test Accuracy: {:.4f}'.format(num_correct/num_examples))