#python img2video.py -vn image
import os
import argparse
import cv2
import numpy as np
from glob import glob
from PIL import Image
from dataloader import img2video, dataLoader_img

parser = argparse.ArgumentParser(description="Reading image like video")
parser.add_argument('-vn', '--video_name', default='/image', type=str)

args = parser.parse_args()
root = args.video_name



def get_frame(video_name):
    images = img2video(video_name)
    for img in images:
        frame = cv2.imread(img)
        yield frame

def get_plenframe(video_name, loc):
    images = dataLoader_img(video_name,loc)
    for img in images:
        frame = cv2.imread(img)
        yield frame

def main():
    for frame in get_frame(root):
        max_size = 960
        if max(frame.shape[:2]) > max_size:
            scale = max_size / max(frame.shape[:2])
            out_size = (
                int(frame.shape[1] * scale),
                int(frame.shape[0] * scale))
            frame = cv2.resize(frame, out_size)
        cv2.imshow('video',frame)
        cv2.waitKey(40)

def main2():
    for frame in get_plenframe(root,'007'):
        max_size = 960
        if max(frame.shape[:2]) > max_size:
            scale = max_size / max(frame.shape[:2])
            out_size = (
                int(frame.shape[1] * scale),
                int(frame.shape[0] * scale))
            frame = cv2.resize(frame, out_size)
        cv2.imshow('video',frame)
        cv2.waitKey(40)

if __name__ == "__main__":
    main()