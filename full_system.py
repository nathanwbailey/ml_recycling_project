import cv2
from PIL import image
import argparse
import numpy as np
from skimage import transform
import time

mean_detect = np.array([0.5250, 0.4508, 0.4446])
std_detect = np.array([0.1288, 0.1237, 0.1117])
mean_classify_network = np.array([0.6661, 0.6211, 0.5492])
std_classify_network = np.array([0.2871, 0.2917, 0.3310])

path_to_video = ''

video = cv2.VideoCapture(path_to_video)
fps_of_video=30
cropped_frames = []

batch_size_detect = 1
batch_size_classify = 1
isProcessingVideo = True

fps_frame_extract = 0
frame_idx = 0


t = time.time()
ret, frame = video.read()
if frame is None:
    isProcessingVideo = False
    fps_frame_extract += (time.time()-t)
if ret:
    cropped_img = frame[(370-110):370, 700:800]
    cropped_frames.append(cropped_img)
    fps_frame_extract += (time.time()-t)
    frame_idx += 1
    
t = time.time()



