import numpy as np
from networking_env.utils.shared_consts import SizeConsts
import tqdm


class HistoryFromText(object):
    HIST_ID = 0

    # TODO create base history file?

    def __init__(self, fname, tm_length_func=lambda: 5, max_steps=10):
        HistoryFromText.HIST_ID += 1

        self._tms = []
        self._tm_times = []
        self._tm_ind = 0
        self._fname = fname

        self._populate(fname, max_steps, tm_length_func)
        self._opts = self._read_opt(fname, max_steps)

    def ttl_time(self):
        # TODO do we get off by one here?
        return sum(self._tm_times)

    def get_next(self):
        tm = self._tms[self._tm_ind]
        tm_time = self._tm_times[self._tm_ind]
        opt_val = self._opts[self._tm_ind] if self._opts else 1.0  # 处理异常opt值？
        self._tm_ind = (self._tm_ind + 1) % len(self._tms)
        return tm, tm_time, opt_val

    def is_over(self):
        return self._tm_ind == len(self._tms)

    def __len__(self):
        return len(self._tms)

    def reset(self):
        self._tm_ind = 0

    def _parse_tm_line(self, line):
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype=np.float64)
        num_nodes = int(np.sqrt(tm.shape[0]))
        tm = tm.reshape((num_nodes, num_nodes))

        return (tm - tm * np.eye(num_nodes)).flatten()


class Histories(object):

    def __init__(self, files_tms, htype, num_nodes, time, files_latent, tm_length_func=lambda: 5, max_steps=60):
        self._tms = []
        self._latent = []
        self._opts = []
        self._tm_times = []
        self._tm_ind = 0
        self._type = htype
        self._max_steps = max_steps
        self._tm_mask = np.ones((num_nodes, num_nodes), dtype=bool).flatten()
        self._tm_mask[np.eye(num_nodes).flatten() == 1] = False
        self._time = time

        # hzb
        self._tms_dict = {}
        self._opts_dict = {}
        # end

        for fname in files_tms:
            # print('[+] Populating TMS.')
            print(f'[+] Populating TMS. {fname}')
            # hzb
            len_tms = len(self._tms)
            len_opts = len(self._opts)
            # end

            self._populate_tms(fname, tm_length_func)
            self._read_opt(fname)

            # hzb
            if htype=='test':
                fni = fname.rfind("\\")
                tm_file_name=fname[fni+1:]
                self._tms_dict[tm_file_name] = self._tms[len_tms:]
                self._opts_dict[tm_file_name] = self._opts[len_opts:]
                # print(self._tms_dict.keys())
            # end

        for fname in files_latent:
            print('[+] Populating latent representation for RL.')
            self._populate_latent(fname, tm_length_func)

    def _read_opt(self, fname):
        try:
            with open(fname.replace('hist', 'opt')) as f:
                lines = f.readlines()
                self._opts += [np.float64(_) for _ in lines]
        except:
            return None

    def _populate_tms(self, fname, tm_length_func):
        """生成流量矩阵tm，和时间序列"""
        with open(fname) as f:
            for line in tqdm.tqdm(f.readlines()):
                try:
                    tm = self._parse_tm_line(line, self._time)
                except:
                    import pdb;
                    pdb.set_trace()

                tm_time = tm_length_func()
                tm = SizeConsts.BPS_TO_GBPS(tm)  # 流量化为GB级别
                self._tms.append(tm)
                self._tm_times.append(tm_time)

    def _populate_latent(self, fname, tm_length_func):
        with open(fname) as f:
            for line in f.readlines():
                latent = self._parse_latent_line(line)
                self._latent.append(latent)

    def __len__(self):
        return len(self._tms)

    def _parse_latent_line(self, line):
        latent = np.array([np.float64(_) for _ in line.split(" ") if _], dtype=np.float64)
        return latent

    def _parse_tm_line(self, line, time):
        # 为什么把0去掉啊 A:去掉自己到自己的
        # 就只看非零流量的情形？
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype=np.float64)

        # the next line removes time features
        if time is True:
            tm = tm[2:]
        num_nodes = int(np.sqrt(tm.shape[0]))
        tm = tm.reshape((num_nodes, num_nodes))

        tm = (tm - tm * np.eye(num_nodes))  # 去除自己到自己的流量
        return tm.flatten()[self._tm_mask]

    def get_next(self):
        tm = self._tms[self._tm_ind]
        tm_time = self._tm_times[self._tm_ind]
        opt_val = self._opts[self._tm_ind]
        return tm, tm_time, opt_val

    def get_next_latent(self):
        latent = self._latent[self._tm_ind]
        tm_time = self._tm_times[self._tm_ind]
        opt_val = self._opts[self._tm_ind]
        if self._max_steps == -1:
            self._tm_ind = (self._tm_ind + 1) % len(self._tms)
        else:
            self._tm_ind = (self._tm_ind + 1) % self._max_steps
        return latent, tm_time, opt_val

    def num_tms(self):
        return len(self._tms)

    def num_histories(self):
        return 1

    def reset(self):
        self._tm_ind = 0
