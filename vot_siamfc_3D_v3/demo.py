#python demo.py --model=../pretrained -vn ../data/NonVideo4 -lf 005 -lfs 27  007, 78

from __future__ import absolute_import

import os
import glob
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from dataloader import dataLoader_img, dataLoader_focal #일반이미지 읽기 focal 이미지 읽기
from dataloader import AllfocalLoader
from siamfc_mine import TrackerSiamFC

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='modelfolder for demo',
                    required=True)
parser.add_argument('-vn','--video_name', default='../../2020VOT_SiamFC/data/NonVideo4_1', type=str,
                    help='videos or image files')
parser.add_argument('-lf','--location', default='005', type=str,
                    help='img location')
parser.add_argument('-lfs','--locations', default='27', type=int,
                    help='img location solution')
#NonVideo4는 lf가 005
#Video3_tiny는 lf가 007

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

#ROI를 지정한 첫번째 이미지를 불러옵니다.
first_window = root + '/000/images/' + lf + '.png' 
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


