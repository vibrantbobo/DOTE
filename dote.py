import sys
import os

cwd = os.getcwd()
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from networking_env.environments.ecmp.env_args_parse import parse_args
from networking_env.environments.ecmp import history_env
from networking_env.environments.consts import SOMode
from networking_env.utils.shared_consts import SizeConsts
from tqdm import tqdm

# dataset definition
class DmDataset(Dataset):
    def __init__(self, props=None, env=None, is_test=None):
        # store the inputs and outputs
        assert props != None and env != None and is_test != None

        num_nodes = env.get_num_nodes()
        env.test(is_test)   # test 什么
        # 历史流量矩阵 维度[8080,22*21]，5个hist文件，每个文件包含2016个流量矩阵
        # 去除自己到自己的流量后，每个流量矩阵的流量剩下 22*22 - 22 = 22*21 = 462
        tms = env._simulator._cur_hist._tms
        # 对应的历史最优??什么 分流后的最优流量矩阵??? 维度[8080]
        # 每个最优方案对应一个流量矩阵，这个最优方案是什么？
        opts = env._simulator._cur_hist._opts
        tms = [np.asarray([tms[i]]) for i in range(len(tms))]
        np_tms = np.vstack(tms)
        np_tms = np_tms.T   # 维度[22*21, 8080]
        # 拉直，维度[]
        # 默认:'C',按行展开; 'F'按列展开
        np_tms_flat = np_tms.flatten('F')   # 按列拉直，毕竟1列就是一个流量矩阵

        assert (len(tms) == len(opts))
        # 划分成H长的样本，类似滑动窗口，所以共有（total_num-H+1）个
        X_ = []
        for histid in range(len(tms) - props.hist_len):
            start_idx = histid * num_nodes * (num_nodes - 1)    # 0
            end_idx = start_idx + props.hist_len * num_nodes * (num_nodes - 1)  # 5554=0+12* 22* (22-1)
            X_.append(np_tms_flat[start_idx:end_idx])

        self.X = np.asarray(X_) # 步长为H的流量需求样本 维度[8080-H, 22*21*H]=[8068,5544]
        # y中的每个元素 y[i] 指的是 流量矩阵tms[i] 与 其对应的最优xx opts[i]
        # 表示，这种流量矩阵情形 与 最优方案的 对应/映射
        self.y = np.asarray([np.append(tms[i], opts[i]) for i in range(props.hist_len, len(opts))])


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition
class NeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxUtil, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

def loss_fn_maxutil(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()
    
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item() # 取出当前流量矩阵所对应的最优方案
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))    # 去除opt

        # 论文里说
        # is training a decision model on past realizations of A
        # and B’s traffic demands to directly output traffic splitting ratios
        # that are close to the global optimum.
        # 也就是说DNN输出一个分流比?

        y_pred = y_pred + 1e-16 #eps    #? 避免非0
        # 这里的权重是啥意思?    A: 如果是分流比
        paths_weight = torch.transpose(y_pred, 0, 1)    # 转置
        # commodities_to_paths[462,1098],节点对i有哪些路径可达，可达的话取值为1
        # 节点对中，其可达路径上的分流比总和（这些可达性是根据流量矩阵计算过的）
        # 所以所有路径的分流比，加起来可能大于1
        commodity_total_weight = commodities_to_paths.matmul(paths_weight)
        # 所以这里? ???? 有些值仍然大于1 只是为了缩小?
        commodity_total_weight = 1.0 / (commodity_total_weight) # 这里为什么
        # 路径上的总分流比 = 这条路径哪些节点对使用到*节点对的分流比总和
        paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)
        # 路径分流比(1098,1) = 预测分流比(1098,1) * 预测路径总分流比(1098,1)
        # 那这里为啥相乘???? 为了再次缩小?
        paths_split = paths_weight.mul(paths_over_total)
        # 可达路径上的流量需求
        tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_true.transpose(0,1))
        # 负载均衡后，每条路径上的流量
        demand_on_paths = tmp_demand_on_paths.mul(paths_split)
        # 每条边上的流量需求
        flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths)
        # 边的利用率（边的流量需求/边的最大容量）
        congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).transpose(0,1))
        # 最大利用率
        max_cong = torch.max(congestion)
        # print(f'max_cong: {max_cong}')
        # print(f'max_cong.item(): {max_cong.item()}')
        loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item()   # 这是什么操作
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt #
        losses.append(loss)
        loss_vals.append(loss_val)
    
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val

class NeuralNetworkMaxFlowMaxConc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxFlowMaxConc, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_dim),
            nn.ELU(alpha=0.1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

def loss_fn_maxflow_maxconc(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()

    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

        y_pred = y_pred + 0.1 #ELU
        edges_weight = paths_to_edges.transpose(0,1).matmul(torch.transpose(y_pred, 0, 1))
        alpha = torch.max(edges_weight.divide(torch.tensor(np.array([env._capacities])).transpose(0,1)))
        max_flow_on_tunnel = y_pred / alpha
        max_flow_per_commodity = commodities_to_paths.matmul(max_flow_on_tunnel.transpose(0,1))

        if props.opt_function == "MAXFLOW": #MAX FLOW
            max_mcf = torch.sum(torch.minimum(max_flow_per_commodity.transpose(0,1), y_true))
            
            loss = -max_mcf if max_mcf.item() == 0.0 else -max_mcf/max_mcf.item()
            loss_val = 1.0 if opt == 0.0 else max_mcf.item()/SizeConsts.BPS_TO_GBPS(opt)
        
        elif props.opt_function == "MAXCONC": #MAX CONCURRENT FLOW
            actual_flow_per_commodity = torch.minimum(max_flow_per_commodity.transpose(0,1), y_true)
            max_concurrent_vec = torch.full_like(actual_flow_per_commodity, fill_value=1.0)
            mask = y_true != 0
            max_concurrent_vec[mask] = actual_flow_per_commodity[mask].divide(y_true[mask])
            max_concurrent = torch.min(max_concurrent_vec)
            
            #actual_flow_per_commodity = torch.minimum(max_flow_per_commodity.transpose(0,1), y_true)
            #actual_flow_per_commodity = torch.maximum(actual_flow_per_commodity, torch.tensor([1e-32]))
            #max_concurrent = torch.min(actual_flow_per_commodity.divide(torch.maximum(y_true, torch.tensor([1e-32])))
            
            loss = -max_concurrent if max_concurrent.item() == 0.0 else -max_concurrent/max_concurrent.item()
            loss_val = 1.0 if opt == 0.0 else max_concurrent.item()/opt
                
            #update concurrent flow statistics
            if concurrent_flow_cdf != None:
                curr_dm_conc_flow_cdf = [0]*len(concurrent_flow_cdf)
                for j in range(env.get_num_nodes() * (env.get_num_nodes() - 1)):
                    allocated = max_flow_per_commodity[j][0].item()
                    actual = y_true[0][j].item()
                    curr_dm_conc_flow_cdf[j] = 1.0 if actual == 0 else min(1.0, allocated / actual)
                curr_dm_conc_flow_cdf.sort()
                
                for j in range(len(curr_dm_conc_flow_cdf)):
                    concurrent_flow_cdf[j] += curr_dm_conc_flow_cdf[j]
        else:
            assert False
        
        losses.append(loss)
        loss_vals.append(loss_val)

    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val
    
# 解析执行参数（flags）
props = parse_args(sys.argv[1:])
env = history_env.ECMPHistoryEnv(props)
# 创建稀疏矩阵
ctp_coo = env._optimizer._commodities_to_paths.tocoo()
# 化为稀疏矩阵Tensor
commodities_to_paths = torch.sparse_coo_tensor(np.vstack((ctp_coo.row, ctp_coo.col)), torch.DoubleTensor(ctp_coo.data), torch.Size(ctp_coo.shape))
# 路径表,元素是用边id表示的路径(edge[5]=1,edge[6]=1，路径包含边5，边6),索引与节点对出现一一对应(env._optimizer._commodities_paths，节点对)
pte_coo = env._optimizer._paths_to_edges.tocoo()
paths_to_edges = torch.sparse_coo_tensor(np.vstack((pte_coo.row, pte_coo.col)), torch.DoubleTensor(pte_coo.data), torch.Size(pte_coo.shape))

batch_size = props.so_batch_size
n_epochs = props.so_epochs
concurrent_flow_cdf = None
if props.opt_function == "MAXUTIL":
    NeuralNetwork = NeuralNetworkMaxUtil
    loss_fn = loss_fn_maxutil
elif props.opt_function == "MAXFLOW":
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc
elif props.opt_function == "MAXCONC":
    if batch_size == 1:
        batch_size = props.so_max_conc_batch_size
        n_epochs = n_epochs*batch_size
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc
    if props.so_mode == SOMode.TEST:
        concurrent_flow_cdf = [0] * (env.get_num_nodes()*(env.get_num_nodes()-1))
else:
    print("Unsupported optimization function. Supported functions: MAXUTIL, MAXFLOW, MAXCOLC")
    assert false

if props.so_mode == SOMode.TRAIN: #train
    # create the dataset
    # 构造一个样本X[8080-H,H*22*21] 22*21是因为去除自己到自己的流量(22个)
    # y 是流量矩阵与最优方案的对应/映射
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    # DataLoader的使用 {https://blog.csdn.net/weixin_45662399/article/details/127601983}
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #create the model
    # 去除了自己到自己的流量，所以DNN输入维度为H*n*(n-1)，输出维度为路径总数
    model = NeuralNetwork(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), env._optimizer._num_paths)
    model.double()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        with tqdm(train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            # targets 维度[batch_size, 463]，为什么是463
            # A: Dataset中y(这里的targets) 构造中是np.append(tms[i], opts[i]) ，流量矩阵和opts合在一起就是463
            for (inputs, targets) in tepoch:    # Dataloader 取出的每个对象中，是一个batch大小的数据
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                yhat = model(inputs)
                # Q: yhat, targets分别是什么
                # yhat: 根据流量矩阵预测出的一个方案，维度[路径数量]
                # targets: batch大小中，每个流量矩阵的最优方案
                loss, loss_val = loss_fn(yhat, targets, env)
                print(f'loss: {loss}')
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss_val)
                loss_sum += loss_val
                loss_count += 1
                loss_avg = loss_sum / loss_count
                tepoch.set_postfix(loss_val=loss_avg)

    #save the model
    torch.save(model, 'model_dote.pkl')

elif props.so_mode == SOMode.TEST: #test
    # create the dataset
    test_dataset = DmDataset(props, env, True)
    # create a data loader for the test set
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #load the model
    model = torch.load('model_dote.pkl')
    model.eval()
    with torch.no_grad():
        with tqdm(test_dl) as tests:
            test_losses = []
            for (inputs, targets) in tests:
                pred = model(inputs)
                test_loss, test_loss_val = loss_fn(pred, targets, env)
                test_losses.append(test_loss_val)
                tests.set_postfix(loss_val=test_loss_val)

            avg_loss = sum(test_losses) / len(test_losses)
            print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
            #print statistics to file
            with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'so_stats.txt', 'w') as f:
                import statistics
                dists = [float(v) for v in test_losses]
                dists.sort(reverse=False if props.opt_function == "MAXUTIL" else True)
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
            
            if concurrent_flow_cdf != None:
                concurrent_flow_cdf.sort()
                with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'concurrent_flow_cdf.txt', 'w') as f:
                    for v in concurrent_flow_cdf:
                        f.write(str(v / len(dists)) + '\n')
# elif props.so_mode == 'my_test':
#     # create the dataset
#     test_dataset = DmDataset(props, env, True)
#     # create a data loader for the test set
#     test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     # load the model
#     model = torch.load('model_dote.pkl')
#     model.eval()
#     with torch.no_grad():
#         with tqdm(test_dl) as tests:
#             test_losses = []
#             for (inputs, targets) in tests:
#                 pred = model(inputs)
#                 test_loss, test_loss_val = loss_fn(pred, targets, env)
#                 test_losses.append(test_loss_val)
#                 tests.set_postfix(loss_val=test_loss_val)
#
#             avg_loss = sum(test_losses) / len(test_losses)
#             print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
