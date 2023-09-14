import os
import random
import warnings

import numpy as np

random.seed(1234)


def read_tms(tm_file):
    tms = []
    with open(tm_file, 'r') as file:
        contents = file.readlines()
        for line in contents:
            tm = line.split()
            tm = list(map(float, tm))
            tms.append(tm)
    return tms


def get_random_index(node_num=22):
    src = int(random.random() * node_num)
    tar = int(random.random() * node_num)
    while src == tar:
        tar = int(random.random() * node_num)
    index = src * node_num + tar
    # print(f'src:{src}, tar:{tar}, index:{index}')
    return index


def get_random_indexes(size=10):
    indexes = []
    for i in range(size):
        indexes.append(get_random_index())
    return indexes


def specify_burst_demand(tm, factor, indexes=None):
    """indexes not None,则对特定位置的需求进行突发，否则对所有需求进行突发"""
    if indexes is None:
        indexes = range(len(tm))
    for i in indexes:
        tm[i] = tm[i] * factor


def burst_tm(tm, random_range=(2, 10)):
    """对tm进行流量突发"""
    indexes = get_random_indexes()
    for i in indexes:
        factor = random.random() * (random_range[1] - random_range[0]) + random_range[0]  # 流量突发(100,200)倍
        tm[i] = tm[i] * factor


# 一个文件共有2016个tm，选择某些tm进行突发流量，或是将所有tm都进行突发
def burst_tms(tms, indexes=None, random_range=(100, 200)):
    """选择indexes里面的流量矩阵进行突发"""
    if indexes == None:
        # burst all
        for tm in tms:
            burst_tm(tm, random_range)
    else:
        for i in indexes:
            burst_tm(tms[i], random_range)


def write_tms(file_name, tms):
    with open(file_name, 'w') as file:
        for tm in tms:
            for i in range(len(tm)):
                if i != 0:
                    file.write(' ')
                file.write(str(tm[i]))
            file.write('\n')


def scale_tm(tm, scale):
    """缩放tm"""
    specify_burst_demand(tm, scale)


def partial_scale_tm(tm, scale, indexes):
    """部分放大tm"""
    specify_burst_demand(tm, scale, indexes)


def random_tm(tm, random_range):
    """
    随机化tm，类似生成新的tm
    :param range: (lower, upper)
    """
    specify_burst_demand(tm, random.random() * (random_range[1] - random_range[0]) + random_range[0])


def scale_tms(tms, scale):
    for i in range(len(tms)):
        scale_tm(tms[i], scale)
        # tms[i] = [t * scale for t in tms[i]]


def gen_scale_date(tms_file, scale):
    tms = read_tms(tms_file)
    scale_tms(tms, scale)
    filename = test_path + tms_file[len(train_path):-5] + f'_scale_{scale}.hist'
    write_tms(filename, tms)
    filenames.append(filename[len(test_path):])
    print(f'{filename} DONE!')


# 三种异常tm，均匀放大，随机化，部分放大
## 均匀放大
def gen_scale_burst_data(tms_file, scale, indexes):
    tms = read_tms(tms_file)
    # assert indexes in range(0, len(tms))  # 不能越界
    for i in indexes:
        scale_tm(tms[i], scale)
    filename = test_path + tms_file[len(train_path):-5] + f'_scale_burst_{scale}.hist'
    write_tms(filename, tms)
    filenames.append(filename[len(test_path):])
    print(f'{filename} DONE!')


## 部分放大
def gen_partial_scale_burst_data(tms_file, scale, indexes):
    tms = read_tms(tms_file)
    # assert indexes in range(0, len(tms))  # 不能越界
    for i in indexes:
        partial_scale_tm(tms[i], scale, get_random_indexes(20))
    filename = test_path + tms_file[len(train_path):-5] + f'_partial_scale_burst_{scale}.hist'
    write_tms(filename, tms)
    filenames.append(filename[len(test_path):])
    print(f'{filename} DONE!')


## 随机化
def gen_random_burst_data(tms_file, random_range, indexes):
    tms = read_tms(tms_file)
    # assert indexes in range(0, len(tms))  # 不能越界
    for i in indexes:
        random_tm(tms[i], random_range)
    filename = tms_file[len(train_path):-5] + f'_random_burst_{random_range[0]}_{random_range[1]}.hist'
    filename = test_path + filename
    write_tms(filename, tms)
    filenames.append(filename[len(test_path):])
    print(f'{filename} DONE!')


filenames = []

test_path = './networking_envs/data/GEANT/test/'
train_path = './networking_envs/data/GEANT/train/'
train_tms_file = '4.hist'

assert os.path.exists(train_path + train_tms_file)
tms_file = train_path + train_tms_file

# H=12, 0-11,12...
# len(file)=2016
indexes = [12] + list(range(100, 2100, 100))
print(indexes)
#[12, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

gen_scale_burst_data(tms_file, 3, indexes)  # 放大
gen_partial_scale_burst_data(tms_file, 3, indexes)  # 部分放大
gen_random_burst_data(tms_file, (0.1, 3), indexes)  # 随机化

# 全局放大
gen_scale_date(tms_file, 2)
gen_scale_date(tms_file, 4)
gen_scale_date(tms_file, 6)
gen_scale_date(tms_file, 8)

print(filenames)
# ['4_scale_burst_3.hist', '4_partial_scale_burst_3.hist', '4_random_burst_0.1_3.hist', '4_scale_2.hist', '4_scale_4.hist', '4_scale_6.hist', '4_scale_8.hist']

# 测试：9.hist,和上述

