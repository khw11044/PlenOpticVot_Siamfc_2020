
#python demo.py --model=../pretrained -vn C:\Repos\vot\2020VOT_SiamFC\data\NonVideo4 -lf 005
from __future__ import absolute_import

import os
import glob
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from dataloader import dataLoader_img, dataLoader_focal #focal 이미지 읽기
from siamfc2D import TrackerSiamFC

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='modelfolder for demo',
                    required=True)
parser.add_argument('-vn','--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('-lf','--locateframe', default='000', type=str,
                    help='select firstframe')


args = parser.parse_args()
modeldir = args.model
root=args.video_name
firstframe=args.locateframe

CWD_PATH = os.getcwd()


if modeldir:
    PATH_TO_DATA = os.path.join(CWD_PATH,modeldir)
    modelpth = glob.glob(PATH_TO_DATA + '/*')[0] #pretrained
    print(modelpth)


#ROI를 지정한 첫번째 이미지를 불러옵니다.
first_window = root + '/000/images/' + firstframe +'.png'   
anno=[]
img = cv2.imread(first_window)

cv2.namedWindow(first_window)
cv2.imshow(first_window, img)
img = cv2.putText(img, str(first_window.split('/')[-1]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
# setting ROI 
rect = cv2.selectROI(first_window, img, fromCenter=False, showCrosshair=True)
anno.append(rect) # left, top, w, h = x, y, w, h  첫번째 꼭지점 (x,y) 두번째 꼭지점 (x+w,y+h)
cv2.destroyWindow(first_window)


if __name__ == '__main__':
    print('start')
    images = dataLoader_img(root,firstframe)   #root는 데이터 폴더 NonVideo4_tiny폴더이고 firstframe : lf 027번째 위치 focal
    show = None
    tracker = TrackerSiamFC(net_path=modelpth)
    tracker.track(img, images, show, anno[0], visualize=True) #img는 첫 선택사진, images는 focal이미지, show는 보여줄 이미지, anno[0]은 첫 bbox

    


    


