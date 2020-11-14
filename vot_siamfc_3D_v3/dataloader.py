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
                    file_root =  root +'/'+dir_name + '/' + locateframe +'.png'
                    #print(" root : " + file_root)
                    filelist.append(file_root)

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

def remove_focal(video_name,locateframe):
    filelist = []
    root_dir = video_name
    for (root, dirs, files) in os.walk(root_dir):
        #print(" root : " + root)
        if len(dirs) > 0:
            for dir_name in dirs:
                #print(" root : " + dir_name)
                if dir_name == 'focal' :
                    file =  root +'/'+dir_name + '/' + locateframe +'.png'
                    os.remove(file)



def listdirLoader(root): 
    files = []
    #root = '../siamfc-pytorch/tools/data/NonVideo4_tiny'
    path = os.listdir(root)
    return path

def AllfocalLoader(root):
    local = []
    for f, frame in enumerate(os.listdir(root)):
        local.append(root + '/' + frame + '/focal') # + '/focal' 
    return local

def AllframeLoader(root):    #모든 프레임 폴더 [NonVideo4/000 001  002  003 .....]
    local = []
    for f, frame in enumerate(os.listdir(root)):
        local.append(root + '/' + frame) # + '/focal' 
    return local

if __name__ == "__main__":
    video_name = 'C:/Repos/vot/2020VOT_SiamFC/vot_My_siamfc/image'
    locateframe ='005'
    print(AllfocalLoader('C:/Repos/vot/2020VOT_SiamFC/data/Video3'))
    #print(img2video(video_name))
    #while True:
    #    ask=str(input('deleteNUM'))
    #    remove_focal('C:/Repos/vot/2020VOT_SiamFC/data/Video3',ask)