from time import time
#from MyDataset import LmdbData
import os
import numpy as np
import re

def parse_dir(parent_dir,sub_dir):
    len_max = 0
    len_min = 1e10
    dir=os.path.join(parent_dir,sub_dir)
    print('---------------------------------------------------')
    print('parsing dir: %s'%dir)
    label=sub_dir
    for file_name in os.listdir(dir):
        file_path=os.path.join(dir,file_name)
        print('parsing file: %s' % file_path)
        data,(mi, ma)=parse_file_new(file_path)
        len_max = max(ma, len_max)
        len_min = min(mi, len_min)
        # for item in data:
        #     lmdbData.add(item,label)
    return data, (len_min, len_max)

def parse_file_new(path):
    with open(path) as f:
        lines = f.readlines()

    point_info_len = 16
    max_points = 0
    min_points = 1e10
    sequence = []
    frame = []

    for i in range(0, len(lines)-point_info_len+1, point_info_len):
        point_id    = int(lines[i+6].split()[1])
        x           = float(lines[i+7].split()[1])
        y           = float(lines[i+8].split()[1])
        z           = float(lines[i+9].split()[1])
        rang        = float(lines[i+10].split()[1])
        velocity    = float(lines[i+11].split()[1])
        doppler     = int(lines[i+12].split()[1])
        bearing     = float(lines[i+13].split()[1])
        intensity   = float(lines[i+14].split()[1])
        if point_id == 0 and i != 0:
            sequence.append(frame)
            max_points = max(max_points, len(frame))
            min_points = min(min_points, len(frame))
            frame = []
        frame.append([x, y, z, rang, velocity, doppler, bearing, intensity])
    sequence.append(frame) # append last frame
    max_points = max(max_points, len(frame))
    min_points = min(min_points, len(frame))   

    data=[]
    window=60
    sliding=10
    frame_id=0
    while frame_id+window<len(sequence):
        data.append(sequence[frame_id:frame_id+window])
        frame_id+=sliding
    print(len(sequence), min_points, max_points)
    return data, (min_points, max_points)


parent_dir = 'data/Train'
parent_dir2 = 'data/Test'
sub_dirs=['boxing','jack','jump','squats','walk']


if __name__=="__main__":

    #lmdbData_train=LmdbData('data/lmdbData_train',map_size = 2)
    #lmdbData_test = LmdbData('data/lmdbData_test_2', map_size=2)
    len_max_min = [0, 1e10]
    start = time()
    # for sub_dir in sub_dirs:
    #     len_mm = parse_dir(parent_dir,sub_dir,lmdbData_train)
    #     len_max_min = [max(len_mm[0], len_max_min[0]), min(len_mm[1], len_max_min[1])]
    # for sub_dir in sub_dirs:
    #     len_mm = parse_dir(parent_dir2,sub_dir,lmdbData_test)
    #     len_max_min = [max(len_mm[0], len_max_min[0]), min(len_mm[1], len_max_min[1])]
    print(len_max_min, time()-start)

