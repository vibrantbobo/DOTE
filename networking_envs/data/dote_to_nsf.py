import math
import os
import pandas as pd
import numpy as np

dataset_path = './GEANT/'

src_topo = "GEANT"
assert os.path.exists(dataset_path)

# 修改文件生成位置

# tar_path = './GEANT_NSF/'
tar_path = './GEANT_burst_NSF/'
TMS_path = 'TMS/'

if not os.path.exists(tar_path):
    os.mkdir(tar_path)

if not os.path.exists(tar_path + TMS_path):
    os.mkdir(tar_path + TMS_path)


def read_tms(src_file):
    with open(src_file, 'r') as f:
        contents = f.readlines()
        tms = []
        for line in contents:
            tm = list(map(float, line.split()))
            tms.append(tm)
    return tms


def gen_topo(src_file, tar_file=None):
    df = pd.read_table(src_file, ',', header=None)
    # print(df.head())
    node_num = df[[0, 1]].max().max() + 1  # 读取0,1列最大值+1: 最大节点数
    # print(df.head())
    lines = df.to_numpy()
    m = np.zeros([node_num, node_num], dtype=np.longlong)
    for line in lines:
        src = line[0]
        dest = line[1]
        cap = line[2]
        # linked = line[3]

        m[src][dest] = cap
        m[dest][src] = cap

    m = m.flatten()
    src_nodes = []
    tar_nodes = []
    for j in range(node_num):
        src_nodes.extend([j] * node_num)
        tar_nodes.extend(list(range(node_num)))
    res_df = pd.DataFrame()
    res_df[0] = src_nodes
    res_df[1] = tar_nodes
    res_df[2] = m

    res_df = res_df[res_df[2] != 0]

    # print(res_df.head())

    edge_num = res_df.shape[0]
    print(f'node_num: {node_num}, edge_num: {edge_num}')
    res_df[3] = 1  # 新增一列
    # df_list = df.values.tolist()

    first_line = str(node_num) + ' ' + str(edge_num) + '\n'
    with open(tar_file, 'w') as f:
        f.write(first_line)

    # print(res_df)

    res_df.to_csv(tar_file, sep=' ', header=None, index=False, mode='a')


def gen_tms(src_file, tar_path, size=3):
    tms = read_tms(src_file)
    if len(tms) < size: size = len(tms)
    for i in range(size):
        node_num = int(math.sqrt(len(tms[i])))
        src_nodes = []
        tar_nodes = []
        for j in range(node_num):
            src_nodes.extend([j] * node_num)
            tar_nodes.extend(list(range(node_num)))

        # print(src_nodes)
        # print(tar_nodes)
        assert len(src_nodes) == len(tar_nodes)

        df = pd.DataFrame()

        df['src'] = src_nodes
        df['dest'] = tar_nodes
        df['bw'] = tms[i]  # 按行展开排列
        # print(df.head())
        df = df.loc[df['src'] != df['dest']]  # 过滤相同节点的行

        commi_num = node_num * (node_num - 1)
        first_line = 'DEMANDS ' + str(commi_num)
        label_demand = ['demand_' + str(i) for i in range(commi_num)]
        df['label'] = label_demand

        df = df[['label', 'src', 'dest', 'bw']]  # 调换列顺序
        print("TM-%d" % i)
        print(df.head())
        print('\n')
        tar_file = tar_path + 'TM-' + str(i)
        with open(tar_file, 'w') as f:
            f.write(first_line + '\n')
        df.to_csv(tar_file, sep=' ', index=False, mode='a')


# 生成拓扑文件
gen_topo(dataset_path + src_topo + "_int.pickle.nnet", tar_path + "topo_NSF.txt")

# 生成50个NSF TM
# gen_tms(dataset_path + 'test/6.hist', tar_path + TMS_path, 100)
# gen_tms(dataset_path + 'test/6_partial_scale_100_for_30_demand_in_specify_tm.hist', tar_path + TMS_path, 100)
