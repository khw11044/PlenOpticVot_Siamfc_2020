# recode폴더의 groundtruth.txt를 읽고 이미지에 바운딩박스를 찍어가며 읽고 read_gt폴더에 저장합니다.
# python test_groundtruth.py -vn C:/Repos/vot/2020VOT_SiamFC/vot_My_siamfc/data/NonVideo4_0
import os
import argparse
import cv2
import numpy as np
from glob import glob
from PIL import Image
from dataloader import dataLoader_img, img2video

parser = argparse.ArgumentParser(description="Reading image like video")
parser.add_argument('-vn', '--video_name', default='C:/Repos/vot/2020VOT_SiamFC/data/NonVideo4_0', type=str) #C:\Repos\vot\2020VOT_SiamFC\vot_My_siamfc\image

args = parser.parse_args()
root = args.video_name

def get_frame1(video_name):
    images = dataLoader_img(video_name,'007') #focal
    for img in images:
        frame = cv2.imread(img)
        yield frame

def get_frame2(video_name):
    images = img2video(video_name) 
    for img in images:
        frame = cv2.imread(img)
        yield frame

f = open("recode/groundtruth3.txt", 'r') #./groundtruth1.txt
def main():
    line=[]
    for i, frame in enumerate(get_frame1(root)):
        # resize img if necessary
        max_size = 960
        count= 0
        if max(frame.shape[:2]) > max_size:
            scale = max_size / max(frame.shape[:2])
            out_size = (
                int(frame.shape[1] * scale),
                int(frame.shape[0] * scale))
            frame = cv2.resize(frame, out_size)
        line = f.readline()
        if not line: break
        x,y,w,h = line.split(',')
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        pt1=(int(x-w/2), int(y-h/2))
        pt2=(int(x+w/2), int(y+h/2))
        k = "%d,%d,%d,%d"%(x,y,w,h)
        frame = cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 2)
        frame = cv2.putText(frame, 'groundtruth : '+str(k), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('video',frame)
        savepath = "./read_gt/{0:0=3d}.png".format(i)
        cv2.imwrite(savepath, frame)
        cv2.waitKey(40)
    f.close()

if __name__ == "__main__":
    main()