# Q&A
1. DNN输入维度为什么是
    H*n*(n-1): 去除了自己到自己的流量

2. DNN输出是什么


3. MAXUTL损失函数的计算过程，是怎么计算的


4. loss_fn_maxutil 参数y_pred_batch, y_true_batch分别代表什么


5. DmDataset中的 self.X 和 self.y 分别表示?
    X中的元素为长为H的流量需求矩阵组成的一条记录。y中的每个元素 y[i] 指的是 流量矩阵tms[i] 与 其对应的最优xx opts[i]。表示，这种流量矩阵情形 与 最优方案的 对应/映射

8. 为什么把流量矩阵中的0去掉啊？_parse_tm_line <- history.py <- simulator <- env


7. history.py中时间(self._tm_times)是干什么的?
   其元素，好像要么添加默认的5，要么添加设置的2或0

9. tm_mask 一般是做什么的? （一个全1的矩阵,拉直，将对角元素改为False(0)）
    用来去除自己到自己的流量

10. 每个最优方案对应一个流量需求矩阵，这个最优方案指的是什么？

11. 流量矩阵应该是node_num*node_num 长度的，为什么代码中是node_num*(node_num-1)
    本来是的，总共有22个节点，那么流量矩阵应该有22*22=484个流量，去除掉自己到自己的流量后，为484-22=22*21=462

12. targets 维度[batch_size, 463]，为什么是463
    Dataset中y(这里的targets) 构造中是np.append(tms[i], opts[i]) ，流量矩阵和opts合在一起就是463

13. 计算loss中 y_pred = y_pred + 1e-16 #eps    #? 避免非0吗

14. commodity_total_weight = 1.0 / (commodity_total_weight) # 节点对总权重取倒数，这里为什么

15. 计算loss中，边的权重是指啥

16. 








