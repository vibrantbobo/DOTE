import glob
import os

gravity_path = './GEANT_DOTE_1'
tms = []
# factor = 10 ** 4  # 扩大系数
# factor = 10 ** 5  # 扩大系数
# factor = 10 ** 6  # 扩大系数
# factor = 10 ** 7  # 扩大系数
# factor = 10 ** 3  # 扩大系数
factor = 1.8 * 10 ** 5
for filename in sorted(glob.glob(gravity_path + '/*.m')):
    tm = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            # line = list(map(float, line))
            line = [float(t) * factor for t in line]
            tm.extend(line)
        print(len(tm))

    tms.append(tm)

print(len(tms))

if not os.path.exists(gravity_path + '/hist/'): os.mkdir(gravity_path + '/hist/')

with open(gravity_path + '/hist/' + 'gravity_scale_1.8_10_5.hist', 'w') as f:
    for tm in tms:
        for i in range(len(tm)):
            if i == 0:
                f.write(str(tm[i]))
            else:
                f.write(' ' + str(tm[i]))
        f.write('\n')
