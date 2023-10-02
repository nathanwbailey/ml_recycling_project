full_system_jetson.py import cv2
import numpy as np
import time
import torch
from network import RecycleNetwork
from detection_network import DetectionNetwork
import torchvision
import sys
from btransforms import batch_transforms

#Code for the full system
#Input a video of frames, grab a frame, input into the detection network, if frame input into the classify network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

model_detect = DetectionNetwork().half().to(device)

model_classify = RecycleNetwork(num_classes=5).half().to(device)




#Computed means and std for the 2 networks
transforms_detect = torchvision.transforms.Compose([batch_transforms.Normalize(mean=[0.6661, 0.6211, 0.5492], std=[0.2871, 0.2917, 0.3310], device=device)])

transforms_resize = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150))])

transforms_classify = torchvision.transforms.Compose([batch_transforms.Normalize(mean=[0.6661, 0.6211, 0.5492], std=[0.2871, 0.2917, 0.3310], device=device)])



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
        with torch.no_grad():
            images = torch.from_numpy(np.array(image_list)).permute(0,3,1,2).to(device)
            images = transforms_resize(images)
            images = images.contiguous().float().div(255)
            images = transforms_detect(images).half()
            predictions = torch.argmax(model_detect(images), dim=1)
            predictions = torch.nonzero(predictions == 1).squeeze().tolist()
        if not isinstance(predictions, list):
            predictions = [predictions]
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
        with torch.no_grad():
            images = torch.from_numpy(image_list).permute(0,3,1,2).to(device)
            images = transforms_resize(images)
            images = images.contiguous().float().div(255)
            images = transforms_classify(images).half()
            predictions = torch.argmax(model_classify(images), axis=1)
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