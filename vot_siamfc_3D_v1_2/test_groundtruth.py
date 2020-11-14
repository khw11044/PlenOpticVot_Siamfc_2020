# recode폴더의 groundtruth와 bbox와 precision과 영상을 읽고 보여줍니다. 평균 거리와 IOU도 보여줍니다.
# python test_groundtruth.py -vn C:/Repos/vot/2020VOT_SiamFC/data/Video3 
import os
import argparse
import cv2
import numpy as np
from glob import glob
from PIL import Image
from dataloader import dataLoader_img
from rect import cal_rect

parser = argparse.ArgumentParser(description="Reading image like video")
parser.add_argument('-vn', '--video_name', default='C:/Repos/vot/2020VOT_SiamFC/data/NonVideo4_0', type=str) #C:\Repos\vot\2020VOT_SiamFC\vot_My_siamfc\image

args = parser.parse_args()
root = args.video_name


def get_frame1(video_name):
    images = dataLoader_img(video_name,'007') #focal
    for img in images:
        frame = cv2.imread(img)
        yield frame


f = open("recode/groundtruth3.txt", 'r') #bboxone.txt  ./groundtruth1.txt
fb = open("recode/bbox3.txt", 'r')
pr = open("recode/precision3.txt",'w')
area = open("recode/rec_area3.txt",'w')
def main():
    line=[]
    total=0
    total_area_acc = 0
    prcision_value = 0
    precision_count_5 = 0
    precision_count_10 = 0
    precision_count_15 = 0
    precision_count_20 = 0
    precision_count_40 = 0
    num=0
    area_acc_val=0
    for i, frame in enumerate(get_frame1(root)):
        num=i
        # resize img if necessary
        max_size = 960
        if max(frame.shape[:2]) > max_size:
            scale = max_size / max(frame.shape[:2])
            out_size = (
                int(frame.shape[1] * scale),
                int(frame.shape[0] * scale))
            frame = cv2.resize(frame, out_size)
        line = f.readline()
        lineb = fb.readline()
        if not line: break
        x,y,w,h = line.split(',')
        xb,yb,wb,hb = lineb.split(',')
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        xb = int(xb)
        yb = int(yb)
        wb = int(wb)
        hb = int(hb)

        rect1 = [x,y,w,h]
        rect2 = [xb,yb,wb,hb]
        area_acc = cal_rect(rect1,rect2)[1]
        
        
        precision = ((x-xb)**2 + (y-yb)**2)**0.5

        if precision <= 40 :
            precision_count_40 += 1
        if precision <= 20 :
            precision_count_20 += 1
        if precision <= 15 :
            precision_count_15 += 1
        if precision <= 10 :
            precision_count_10 += 1
        if precision <= 5 :
            precision_count_5 += 1



        total += precision
        pt1=(int(x-w/2), int(y-h/2))
        pt2=(int(x+w/2), int(y+h/2))
        k = "%d,%d,%d,%d"%(x,y,w,h)
        ptb1=(int(xb-wb/2), int(yb-hb/2))
        ptb2=(int(xb+wb/2), int(yb+hb/2))
        kb = "%d,%d,%d,%d"%(xb,yb,wb,hb)
        if i == 0:
            prcision_value == 0
            area_acc_val = area_acc
        else :
            prcision_value = total/i
            total_area_acc += area_acc
            area_acc_val = total_area_acc / i
        print('-------------------------------{}-------{}-------------'.format(area_acc_val,i))
        frame = cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 2)
        frame = cv2.rectangle(frame, ptb1, ptb2, (0, 0, 255), 2)
        frame = cv2.putText(frame, 'groundtruth : '+str(k), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        frame = cv2.putText(frame, 'bbox : ' + str(kb), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = cv2.putText(frame, 'Now IOU : ' + str(area_acc), (200, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        frame = cv2.putText(frame, 'Total IOU : ' + str(area_acc_val), (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        frame = cv2.putText(frame, 'Now_precision : ' + str(precision), (600, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        frame = cv2.putText(frame, 'accuration : ' + str(prcision_value), (600, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow('video',frame)
        savepath = "./test_img/{0:0=3d}.png".format(i)
        cv2.imwrite(savepath, frame)
        area_acc = "%s\n"%area_acc
        area.write(area_acc)
        precision = "%s\n"%precision
        pr.write(precision)
        cv2.waitKey(40)
    f.close()
    pr.close

    print(total,num)
    print('precision_count_5 :', precision_count_5)
    print('precision_count_10 :', precision_count_10)
    print('precision_count_15 :', precision_count_15)
    print('precision_count_20 :', precision_count_20)
    print('precision_count_40 :', precision_count_40)
    print('mean_precision :', prcision_value)
    print('IOU :',area_acc_val)

if __name__ == "__main__":
    main()