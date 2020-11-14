import os
import argparse

import cv2
import numpy as np
from glob import glob
from PIL import Image

def img2video(video_name):
    filelist = []
    root_dir = video_name
    for (root, dirs, files) in os.walk(root_dir):
        print("root : " + root)
        if len(files) > 0:
            for file_dir in files:
                file_root = root + '/' + file_dir
                #print("file root :" + file_root)
                filelist.append(file_root)
    return filelist

def dataLoader_img(video_name,locateframe):
    filelist = []
    root_dir = video_name
    for (root, dirs, files) in os.walk(root_dir):
        if len(dirs) > 0:
            for dir_name in dirs:
                if dir_name == 'images' :
                    file =  root +'/'+dir_name + '/' + locateframe +'.png'
                    #print(" root : " + file)
                    filelist.append(file)

    return filelist

def dataLoader_focal(video_name,locateframe):
    filelist = []
    root_dir = video_name
    for (root, dirs, files) in os.walk(root_dir):
        #print(" root : " + root)
        if len(dirs) > 0:
            for dir_name in dirs:
                #print(" root : " + dir_name)
                if dir_name == 'focal' :
                    file =  root +'/'+dir_name + '/' + locateframe +'.png'
                    #print(" root : " + file)
                    filelist.append(file)
    return filelist

if __name__ == "__main__":
    video_name = '../siamfc-pytorch/tools/data/NonVideo4_tiny'
    locateframe ='005'
    #dataLoader(locateframe)
    print(dataLoader_img(video_name,locateframe))