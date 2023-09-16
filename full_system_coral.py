import cv2
from PIL import Image
import argparse
import numpy as np
from skimage import transform
import time
import tflite_runtime.interpreter as tflite

#Code for the full system
#Input a video of frames, grab a frame, input into the detection network, if frame input into the classify network

model_detect = tflite.Interpreter(model_path='model_detection_edgetpu.tflite', experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
model_detect.allocate_tensors()

model_classify = tflite.Interpreter(model_path='model_classify_edgetpu.tflite',experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
model_classify.allocate_tensors()

#Computed means and std for the 2 networks
mean_detect = np.array([0.5250, 0.4508, 0.4446])
std_detect = np.array([0.1288, 0.1237, 0.1117])
mean_classify = np.array([0.6661, 0.6211, 0.5492])
std_classify = np.array([0.2871, 0.2917, 0.3310])

path_to_video = 'recycling_video2.mp4'

video = cv2.VideoCapture(path_to_video)
fps_of_video=30
cropped_frames = []
cropped_frames_to_classify_network = []

batch_size_detect = 1
batch_size_classify = 1
isProcessingVideo = True

fps_frame_extract = 0
frame_idx = 0

fps_detect = 0
num_detect = 0

fps_classify = 0
num_classify = 0

idx_1 = 0
idx_2 = 0
idx_3 = 0

ex1=True

#function to pass the data through the TF lite model
def pass_through_model(model, data):
    out_data = []
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    for i in data:
        model.set_tensor(input_details[0]['index'], np.expand_dims(i, 0))
        model.invoke()
        query_features = model.get_tensor(output_details[0]['index'])
        out_data.append(np.squeeze(query_features))
    return np.array(out_data)

while isProcessingVideo or len(cropped_frames) > 0 or len(cropped_frames_to_classify_network) > 0:
    t = time.time()
    ret, frame = video.read()
    if frame is None:
        isProcessingVideo = False
        fps_frame_extract += (time.time()-t)
    if ret:
        cropped_img = frame[(370-110):370, 700:800]
        cropped_frames.append(cropped_img)
        if ex1:
            if idx_1 > 0:
                fps_frame_extract += (time.time()-t)
                frame_idx += 1
        else:
            fps_frame_extract += (time.time()-t)
            frame_idx += 1   
        idx_1 +=1
    if len(cropped_frames) >= batch_size_detect or (not isProcessingVideo and len(cropped_frames) > 0):
        t = time.time()
        b_size = batch_size_detect if (isProcessingVideo or len(cropped_frames) > batch_size_detect) else len(cropped_frames)
        image_list = np.array(cropped_frames[:b_size])
        cropped_frames = cropped_frames[b_size:]
        images_resized = []
        for idx, i in enumerate(image_list):
            image = cv2.resize(i, (150, 150))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_resized.append(image)
        img = np.array(images_resized)
        #The processing that we do in pytorch scales the images to [0, 1.0] and then normalizes using the std and mean, so we do that here
        img = np.divide(img, 255)
        img = np.subtract(img, mean_detect)
        img = np.divide(img, std_detect)
        img = img.astype(np.float32)

        detect_model_output = pass_through_model(model_detect, img)

        predictions = np.argmax(detect_model_output, axis=1)
        predictions = np.nonzero(predictions == 1)[0].tolist()
        #cropped_frames_to_classify_network.extend([image_list[i].copy() for i in predictions])
        cropped_frames_to_classify_network.extend(image_list)
        # print(cropped_frames_to_classify_network)
        if ex1:
            if idx_2 > 0:
                fps_detect += (time.time()-t)
                num_detect += len(image_list)
        else:
            fps_detect += (time.time()-t)
            num_detect += len(image_list)
        idx_2 += 1
    

    if len(cropped_frames_to_classify_network) >= batch_size_classify or (not isProcessingVideo and len(cropped_frames_to_classify_network) > 0):
        t = time.time()
        b_size = batch_size_classify if (isProcessingVideo or len(cropped_frames_to_classify_network) > batch_size_classify) else len(cropped_frames_to_classify_network)
        image_list = np.array(cropped_frames_to_classify_network[:b_size])
        cropped_frames_to_classify_network = cropped_frames_to_classify_network[b_size:]
        images_resized = []
        for idx, i in enumerate(image_list):
            image = cv2.resize(i, (150, 150))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_resized.append(image)
        img = np.array(images_resized)
        #The processing that we do in pytorch scales the images to [0, 1.0] and then normalizes using the std and mean, so we do that here
        img = np.divide(img, 255)
        img = np.subtract(img, mean_classify)
        img = np.divide(img, std_classify)
        img = img.astype(np.float32)
        output_classify_network = pass_through_model(model_classify, img)
        predictions = np.argmax(output_classify_network, axis=1)
        if ex1:
            if idx_3 > 0:
                fps_classify += (time.time()-t)
                num_classify += len(image_list)
        else:
            fps_classify += (time.time()-t)
            num_classify += len(image_list)
        idx_3 += 1
                    
video.release()
cv2.destroyAllWindows()
fps_frame_extract = fps_frame_extract / frame_idx
fps_detect = fps_detect / num_detect
fps_classify = fps_classify / num_classify
print(fps_frame_extract)
print(fps_detect)
print(fps_classify)
print('FPS is: {:.0f}FPS'.format(1/(fps_frame_extract+fps_detect+fps_classify)))