import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read logs
logs_path = './REPORT/0927_report/logs/'
logs_path = './REPORT/0925_test_report-log/GEANT_burst/'
filenames = [
    # logs_path + 'model_dote_GEANT_SHRINK_5_gravity_log.txt'
    # ,
    # logs_path + 'model_dote_GEANT_SHRINK_5_log.txt'
    logs_path + 'model_dote_GEANT_log.txt'
]
# for fn in glob.glob(logs_path + '/*.txt'):
for fn in filenames:
    vals = []
    with open(fn, 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            max_cong = float(line[1])
            opt = float(line[2])
            loss_val = float(line[3])
            vals.append([max_cong, opt, loss_val])
        # print(vals)
        df = pd.DataFrame(vals)

        print(f"total vals: {df.shape[0]}")

        b1_1 = df[df[2] > 1.1]
        print(f"loss_val > 1.1: {b1_1.shape[0]}")

        b1_2 = df[df[2] > 1.2]
        print(f"loss_val > 1.2: {b1_2.shape[0]}")

        b1_3 = df[df[2] > 1.3]
        print(f"loss_val > 1.3: {b1_3.shape[0]}")

        b1_4 = df[df[2] > 1.4]
        print(f"loss_val > 1.4: {b1_4.shape[0]}")
