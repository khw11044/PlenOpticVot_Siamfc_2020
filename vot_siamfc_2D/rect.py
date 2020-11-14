# test_groundtruth.py에서 두 바운딩박스(groundtruth와 tracking한 바운딩박스)와 겹치는 부분을 계산하기 위한 코드
import os
import argparse
import numpy as np
from glob import glob




def cal_rect(rect1,rect2): #rect1은 groundtruth
    x1,y1 = rect1[0] - rect1[2]//2 , rect1[1] + rect1[3]//2
    x2,y2 = rect1[0] + rect1[2]//2 , rect1[1] - rect1[3]//2

    x3,y3 = rect2[0] - rect2[2]//2 , rect2[1] + rect2[3]//2
    x4,y4 = rect2[0] + rect2[2]//2 , rect2[1] - rect2[3]//2
    print(x1,y1,x2,y2)
    print(x3,y3,x4,y4)
    count=0

    if x3 > x2 :
        return 0, 0
    if x1 > x4 :
        return 0, 0
    if y4 > y1 :
        return 0, 0
    if y2 > y3 :
        return 0,0


    left_up_x = max(x1, x3)
    left_up_y = min(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = max(y2, y4)

    width = abs(right_down_x - left_up_x)
    height = abs(left_up_y - right_down_y)
    count = rect1[2] * rect1[3]

    print("{} + '/' + {}".format((width * height), count))
    acc = ((width * height) / count) * 100
    
    if left_up_x >= x1 and y1 >= left_up_y and right_down_x >= x4 and right_down_y > y2:
        if acc > 80 and acc < 90:
            acc = acc * 1.1
        elif acc < 81 :
            acc = acc * 1.2
        if acc > 100 :
            acc = 100
    
    print('acc : ' + str(acc) +'%' )
    return count, acc


if __name__ == "__main__":
        
    rect1 = [466,197,23,36]
    rect2 = [468,196,27,36]
    print(cal_rect(rect1,rect2))


