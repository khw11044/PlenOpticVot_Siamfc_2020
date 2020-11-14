#conda activate pytorchsiamfc
#python demo.py --model=../pretrained -vn ../../2020VOT_SiamFC/data/NonVideo4_0 -lf 005 -lfs 27, 50, 79

from __future__ import absolute_import

import os
import glob
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from dataloader import dataLoader_img, dataLoader_focal 
from dataloader import AllfocalLoader
from siamfc_mine import TrackerSiamFC

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='modelfolder for demo',
                    required=True)
parser.add_argument('-vn','--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('-lf','--location', default='005', type=str,
                    help='img location solution')
parser.add_argument('-lfs','--locations', default='027', type=int,
                    help='img location solution')

args = parser.parse_args()
modeldir = args.model
root=args.video_name
lf = args.location
lfs = args.locations

CWD_PATH = os.getcwd()

if modeldir:
    PATH_TO_DATA = os.path.join(CWD_PATH,modeldir)
    modelpth = glob.glob(PATH_TO_DATA + '/*')[0] #pretrained
    print(modelpth)


first_window = root + '/000/images/' + lf +'.png' #007.png
anno=[]
img = cv2.imread(first_window)
img = cv2.putText(img, str(first_window.split('/')[-1]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.namedWindow(first_window)
cv2.imshow(first_window, img)

    
# setting ROI
rect = cv2.selectROI(first_window, img, fromCenter=False, showCrosshair=True)
anno.append(rect)                                        
print(anno[0])
cv2.destroyWindow(first_window)


if __name__ == '__main__':
    print('start')                                             
    show = dataLoader_img(root,lf)                            
    focals = AllfocalLoader(root)                             
    tracker = TrackerSiamFC(net_path=modelpth)
    tracker.track(img, focals, show ,anno[0], lfs, visualize=False)  


