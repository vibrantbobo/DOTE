import os
import random
import numpy as np

base_path = './networking_envs/data/GEANT/test/'
src_tm_file = base_path + '6.hist'

assert os.path.exists(src_tm_file)

gen_tm_file = base_path + '10.hist'

tms = []
random.seed(1234)
with open(src_tm_file, 'r') as file:
    contents = file.readlines()
    for line in contents:
        tm = line.split()
        # factor = random.random() * 1.5 + 0.5  # range(0.5,2)
        # factor = random.random() * 9.9 + 0.1  # range(0.1,10)
        factor = random.random() * 99.9 + 0.1  # range(0.1,10)
        print(factor)
        tm = [float(t) * factor for t in tm]
        tms.append(tm)

with open(gen_tm_file, 'w') as file:
    for tm in tms:
        for i in range(len(tm)):
            if i != 0:
                file.write(' ')
            file.write(str(tm[i]))
        file.write('\n')
