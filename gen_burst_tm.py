import os
import random
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
    print(f'src:{src}, tar:{tar}, index:{index}')
    return index


def get_random_indexes(size=10):
    indexes = []
    for i in range(size):
        indexes.append(get_random_index())
    return indexes


def burst_tm(tm, range=(100, 200)):
    """对tm进行流量突发"""
    indexes = get_random_indexes()
    for i in indexes:
        factor = random.random() * (range[1] - range[0]) + range[0]  # 流量突发(100,200)倍
        print(f'tm before burst: {tm[i]}')
        tm[i] = tm[i] * factor
        print(f'tm after burst: {tm[i]}')


# 一个文件共有2016个tm，选择某些tm进行突发流量，或是将所有tm都进行突发
def burst_tms(tms, indexes=None, range=(100, 200)):
    """选择indexes里面的流量矩阵进行突发"""
    if indexes == None:
        # burst all
        for tm in tms:
            burst_tm(tm, range)
    else:
        for i in indexes:
            burst_tm(tms[i], range)


def write_tms(file_name, tms):
    with open(file_name, 'w') as file:
        for tm in tms:
            for i in range(len(tm)):
                if i != 0:
                    file.write(' ')
                file.write(str(tm[i]))
            file.write('\n')


def scale_tms(tms, scale):
    for tm in tms:
        tm = [t * scale for t in tm]



base_path = './networking_envs/data/GEANT/test/'
src_tm_file = base_path + '6.hist'

assert os.path.exists(src_tm_file)

if __name__ == '__main__':
    # gen_tm_file = base_path + '11.hist'
    # tms = read_tms(src_tm_file)
    # burst_tms(tms)
    # write_tms(gen_tm_file, tms)

    # bust part of tms
    # 一个文件共有2016个tm，选择某些tm进行突发流量
    indexes = list(map(int, np.random.random_sample(200) * 2016))
    # print(indexes)
    gen_tm_file = base_path + '13.hist'
    tms = read_tms(src_tm_file)
    burst_tms(tms, indexes, (0.1, 10))
    write_tms(gen_tm_file, tms)
