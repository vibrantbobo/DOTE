import glob
import os
import random


def read_tms(src_file):
    with open(src_file, 'r') as f:
        contents = f.readlines()
        tms = []
        for line in contents:
            tm = list(map(float, line.split()))
            tms.append(tm)
    return tms


def write_tms(tar_file, tms):
    with open(tar_file, 'w') as f:
        for tm in tms:
            for i in range(len(tm)):
                if i > 0: f.write(' ')
                f.write(str(tm[i]))
            f.write('\n')
    print(f'write to {tar_file} successful!')


def random_tm(tm, factor_range):
    for i in range(len(tm)):
        tm[i] = tm[i] * (random.random() * (factor_range[1] - factor_range[0]) + factor_range[0])


def gen_all_random_data(src_file, tar_file, factor):
    tms = read_tms(src_file)
    for i in range(len(tms)):
        random_tm(tms[i], factor)
    write_tms(tar_file, tms)


def gen_specify_random_data(src_file, tar_file, factor, indexes):
    tms = read_tms(src_file)
    for i in indexes:
        if i >= len(tms):
            print(f'index: {i} overflow')
            continue
        # 合法
        random_tm(tms[i], factor)
    write_tms(tar_file, tms)
    print(f'gen {tar_file} successful!')


def scale_tm(tm, scale):
    for i in range(len(tm)):
        tm[i] = tm[i] * scale


def gen_all_scale_data(src_file, tar_file, scale):
    tms = read_tms(src_file)
    for i in range(len(tms)):
        scale_tm(tms[i], scale)
    write_tms(tar_file, tms)
    print(f'gen {tar_file} successful!')


def gen_specify_scale_data(src_file, tar_file, scale, indexes):
    tms = read_tms(src_file)
    for i in indexes:
        if i >= len(tms):
            print("indexes is unvalid.")
            continue
        print(tms[i])
        scale_tm(tms[i], scale)
    write_tms(tar_file, tms)
    print(f'gen {tar_file} successful!')


def partial_scale_tm(tm, scale, indexes):
    for i in indexes:
        if i >= len(tm):
            continue
        tm[i] = tm[i] * scale


def gen_specify_tm_partial_scale_demand(src_file, tar_file, scale, tm_indexes, d_indexes):
    tms = read_tms(src_file)
    for i in tm_indexes:
        if i >= len(tms):
            continue
        partial_scale_tm(tms[i], scale, d_indexes)
    write_tms(tar_file, tms)


def gen_random_demand_indexes(node_num, size):
    indexes = []
    for i in range(size):
        a = int(random.random() * node_num)
        b = int(random.random() * node_num)
        while a == b:
            b = int(random.random() * node_num)
        indexes.append(a * node_num + b)
    return indexes


def partial_random_tm(tm, factor_range, indexes):
    for i in indexes:
        if i >= len(tm):
            continue
        factor = (random.random() * factor_range[1] + factor_range[0])
        # print(factor)
        tm[i] = tm[i] * factor


def gen_specify_tm_partial_random_demand(src_file, tar_file, factor_range, tm_indexes, d_indexes):
    tms = read_tms(src_file)
    for i in tm_indexes:
        if i >= len(tms):
            continue
        partial_random_tm(tms[i], factor_range, d_indexes)
        # print('\n')
    write_tms(tar_file, tms)


def GEANT_shrink(src_path, factor):
    for d in ['train', 'test']:
        for hist_file in sorted(glob.glob(src_path + d + '/*.hist')):
            print(f'processing {hist_file}')
            tms = read_tms(hist_file)
            for i in range(len(tms)):
                scale_tm(tms[i], factor)
            # print(tms)
            write_tms(hist_file[:-4] + f"shrink_{factor}.hist", tms)


# base_path = "./GEANT/"
# base_path = "./GEANT_SHRINK/"
base_path = "./Abilene/"
train_path = base_path + 'train/'
test_path = base_path + 'test/'
# hist_file = '6.shrink_0.25.hist'
# hist_file = '6.hist'
hist_file = '4.hist'
src_file = test_path + hist_file

scale = 20
indexes = [12, 13, 14, 100, 101, 102, 103, 200, 201, 202]  # 稍微连续的试试
# indexes = [12, 13, 14]  # 稍微连续的试试
# factor_range = (0.1, 2)
# factor_range = (2, 5)
factor_range = (1, 2)

# gen_all_scale_data(src_file, test_path + f'5_all_scale_{scale}.hist', scale)
# gen_specify_scale_data(src_file, test_path + f'4_specify_scale_{scale}.hist', scale, indexes)
#
# gen_all_random_data(src_file, test_path + f'7_all_random_{factor_range[1]}.hist', factor_range)
# gen_specify_random_data(src_file, test_path + f'6_specify_random_{factor_range[1]}.hist', factor_range, indexes)
#
random.seed(1234)
demand_num = 100
demand_indexes = gen_random_demand_indexes(22, demand_num)
# print(demand_indexes)
# gen_specify_tm_partial_scale_demand(src_file,
#                                     test_path + f'{hist_file[0:-4]}_partial_scale_{scale}_for_{demand_num}_demand_in_specify_tm.hist',
#                                     scale,
#                                     indexes,
#                                     demand_indexes)

# gen_specify_tm_partial_random_demand(src_file,
#                                      test_path + f'6_partial_random_{factor_range[0]}_{factor_range[1]}_for_{demand_num}_demand_in_specify_tm.hist',
#                                      factor_range, indexes, demand_indexes)

# GEANT_shrink('./GEANT_SHRINK/', 0.125)
# GEANT_shrink('./GEANT_SHRINK/source/', 0.2)
