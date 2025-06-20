import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch.nn.functional as F
import random
import torch
import pdb
import torch.nn as nn
import sys 
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_dir)
from DLWF_pytorch.model.model_1000 import * 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = nn.CrossEntropyLoss()


# # #trace shape [Ncity,2], the shape of old_trace is [batch_size,200,1],返回值[batch_size,200,1]
def generate_adv_trace(trace, old_trace):

    # 提前检查 trace 是否为空
    if trace.numel() == 0:
        return old_trace

    # 在 CPU 上对插入位置排序
    insert_loc = torch.argsort(trace[:, 0].cpu()).to(trace.device)
    adv_trace = trace[insert_loc]

    # 向量化插入操作
    insert_positions = adv_trace[:, 0].long()
    insert_counts = adv_trace[:, 1].long()
    
    # 计算总插入偏移
    total_insert = insert_counts.sum().item()
    if total_insert == 0:
        return old_trace

    # 计算新 trace 的 shape
    batch_size, seq_len, _ = old_trace.shape
    new_seq_len = seq_len + total_insert
    new_trace = torch.zeros((batch_size, new_seq_len, 1), device=old_trace.device)
    prev_pos = 0
    pos_offset = 0

    # 逐步插入
    for i in range(len(insert_positions)):
        pos = insert_positions[i].item()
        insert_num = insert_counts[i].item()
        # 复制当前 segment
        new_trace[:, prev_pos+pos_offset:pos+pos_offset, :] = old_trace[:, prev_pos:pos, :]

        # 插入元素
        if i == 0:
            insert_val = old_trace[:, pos, :]
        else:
            insert_val = new_trace[:, pos, :]
        
        new_trace[:,pos+pos_offset: pos+pos_offset + insert_num, :] = insert_val.unsqueeze(1).expand(-1, insert_num, -1)
        prev_pos = pos 
        pos_offset += insert_num
    # 复制剩余部分
    new_trace[:, prev_pos+pos_offset:, :] = old_trace[:, prev_pos:, :]
    return new_trace[:,:seq_len,:]


class Ant():
    def __init__(self, model, numant, max_insert, patches,itermax=10):
        self.numant = numant
        self.max_insert = max_insert
        self.patches = patches
        self.model = model
        self.adv_trace = None    
        self.itermax = itermax


    def sensitive_generate(self, test_traces, select_p_num):
        """
        向量化进行敏感性分析,返回每个代表位置插入一个包的对抗序列。
        test_traces: shape = (5000, 5000) 原始测试序列
        select_p_num: int 选择这么多个敏感位置
        """

        batch_size, seq_length = test_traces.shape
        positions_to_test = sorted(random.sample(range(seq_length), select_p_num))

        perturbed_traces = np.zeros((select_p_num, batch_size, seq_length + 1), dtype=test_traces.dtype)

        for i, pos in enumerate(positions_to_test):
            perturbed_traces[i, :, :pos] = test_traces[:, :pos]
            perturbed_traces[i, :, pos] = 1  # 注入虚拟包，方向固定为 +1
            perturbed_traces[i, :, pos+1:] = test_traces[:, pos:]

        return perturbed_traces[:,:,:seq_length], positions_to_test  # shape = (select_p_num, batch, 5000)
    
    def fitness(self, logits, ground_truth):
        loss = F.cross_entropy(logits, ground_truth, reduction='mean')
        return loss.item()
        
    
    def sensitive_results(self,test_traces,ground_truth):
        """
        向量化进行敏感性分析,返回每个代表位置插入一个包的对抗序列。
        test_traces: shape = (5000, 5000) 原始测试序列
        ground_truth: shape = (5000,) 原数据集真实标签集合
        """
        fit_results = []
        
        
        perturbed_traces, positions_to_test = self.sensitive_generate(test_traces,100)
        for i in range(len(positions_to_test)):
            logits = self.model(perturbed_traces[i])
            fit_results.append(self.fitness(logits,ground_truth))
        sorted_index = sorted(range(len(fit_results)),key = lambda i:fit_results[i],reverse=True)
        return positions_to_test[sorted_index]
        
            
            
        


    def find_next_citySolution_fast(self,probtrans):
        """
        使用 torch.multinomial() 直接进行批量概率采样，避免重复计算。
        """
        # 确保概率归一化
        probtrans = probtrans / probtrans.sum()
        # 生成 Ncity 个随机采样索引
        indices = torch.multinomial(probtrans.flatten(), num_samples=self.patches, replacement=False)
        
        # 将 1D 索引转换为 2D 坐标
        cities = torch.div(indices, probtrans.size(1), rounding_mode='trunc')
        insert_packets = indices % probtrans.size(1)
        # print(cities,"\n",insert_packets)

        return cities, insert_packets



    #批处理方式old_trace shape [batch_size,1,200], ground_truth shape [batch_size,100]
    def run(self,old_trace,ground_truth,alpha=0.1,rho=0.15):#gamma调整奖励函数中损失函数的占比
        # print("----------------Ant-algorithm------------")
        numcity = max(old_trace.shape) ##// 城市个数
        iter = 0
        global_pheromonetable = torch.ones((self.numant, numcity, self.max_insert+1))/2 #// 信息素矩阵
        sum_pheromone = torch.ones(numcity,self.max_insert+1)/3
        global_pathtable = torch.zeros((self.numant, self.patches, 2))
        numant_index = torch.arange(self.numant).to(device)
        lengthbest = torch.zeros(1).to(device) #// 最佳路径对应的值
        adv_loc_best = torch.zeros((self.patches,2)) #// 最佳路径
        global_changepheromonetable = torch.zeros((self.numant,numcity,self.max_insert+1))
        
        while (iter < self.itermax) and (sum_pheromone.sum() < 15000):
    
            global_increase_loss = torch.zeros(self.numant,dtype=torch.float).to(device)  # 计算各个蚂蚁的路径距离 
            probtrans = (sum_pheromone ** alpha)
            # * ((8-charu_range[None, :]) ** beta)
        

        # 创建线程池并提交任务
            with ThreadPoolExecutor(max_workers=self.numant) as executor:
                futures = [executor.submit(self.ant_task,probtrans,old_trace,ground_truth,numcity) for i in range(self.numant)]
                # 收集每个线程的返回结果
                count = 0
                for future in as_completed(futures):
                    result = future.result()
                    global_increase_loss[count] = result[0]
                    global_pathtable[count] = result[1]
                    global_changepheromonetable[count] = result[2]
                    count = count+1
            # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数
            if global_increase_loss.max() > lengthbest:
                lengthbest = global_increase_loss.max()
                adv_loc_best = global_pathtable[global_increase_loss.argmax()].clone()
            
            global_pheromonetable[numant_index] = (1 - rho) * global_pheromonetable[numant_index] + global_changepheromonetable[numant_index]  # 计算信息素公式
            sum_pheromone = global_pheromonetable.sum(dim=0)/self.numant
            iter += 1  # 迭代次数指示器+1
            if global_increase_loss.max() > old_trace.shape[0]*0.9:
                break

        return adv_loc_best

    # 定义线程任务函数，每个任务会用到全局变量 global_table，但不能修改它
    def ant_task(self,probtrans,old_trace,ground_truth,numcity):

        pathtable = torch.zeros((self.patches, 2),dtype=torch.int) #// 路径记录表
        cities, insert_packets = self.find_next_citySolution_fast(probtrans)
        pathtable[:, 0] = cities
        pathtable[:, 1] = insert_packets
        adv_trace = generate_adv_trace(pathtable,old_trace)
        # 构造好流之后计算模型loss
        predict = self.model(adv_trace)
        # print(adv_trace.shape)
        batch_acc = (predict.argmax(-1)==ground_truth.argmax(-1)).sum().float()

        adv_num = predict.shape[0]-batch_acc
        # 信息素更新向量化计算
        changepheromonetable = torch.zeros((numcity, self.max_insert + 1), device=probtrans.device)
        
        changepheromonetable[cities, insert_packets] += (adv_num.cpu())/predict.shape[0]

        return adv_num,pathtable,changepheromonetable




