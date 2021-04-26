import os
import io
import sys
import time
import datetime
import subprocess
import argparse

import numpy as np
import cv2

from utils import *

# 1. Command for inference on GPU
# python predict.py -p ./testimage_folder --model_path ./models/mobilenetV2_model --gpu -1 

# 2. Command for inference on CPU
# python predict.py -p ./testimage_folder --model_path ./models/mobilenetV2_model 

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--data_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--gpu', type=int, default=-1, required=False)
parser.add_argument('--biggest_side', type=int, default=0, required=False)
parser.add_argument('--delay', type=int, default=7, required=False)
parser.add_argument('--frame_rate', type=int, default=12, required=False)
parser.add_argument('--denoise_borders', action='store_true')
args = parser.parse_args()
globals().update(vars(args))

biggest_side = None if not biggest_side else biggest_side
delay = round(100/frame_rate + .5)

trainer = Trainer(path=model_path, gpu=gpu) 
if gpu < 0:
    torch.set_num_threads(2)
trainer.load_state(mode="metric")
trainer.model.eval()


print("Check Data_path=",data_path)
files_list = os.listdir(data_path)
print("Check files_list=",files_list)

images = []
if files_list:
    for fname in files_list:   
        if fname.split(".")[-1] in ["jpg","png","jpeg"]:
            images.append(fname)
        
    if images:
        for fname in images:
            img = cv2.imread(data_path+"/"+fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.uint8)
            print("Starting Inference on image = ",fname)
            out = trainer.predict_mask(img, biggest_side=biggest_side, denoise_borders=denoise_borders)
            cv2.imwrite('%s/%s_seg.png' % (data_path, fname.split(".")[0]), out[0])
        print(" [INFO] Images processed! ")

#print(" [INFO] %s ms. " % round((time.time()-start)*1000, 0))