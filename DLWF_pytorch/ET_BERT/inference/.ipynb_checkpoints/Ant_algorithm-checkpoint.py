import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch.nn.functional as F
import random
import torch
import pdb
import torch.nn as nn
import sys 
from tqdm import tqdm
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_dir)

loss = nn.CrossEntropyLoss()

batch_size = 32


class HAAD():
    def __init__(self, model, original_trace, numant, max_inject, max_iter, alpha=1, beta=1, rho=0.1):
        self.numant = numant
        self.max_inject = max_inject
        self.max_iter = max_iter
        self.original_trace = original_trace.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.positions = None   #按照敏感点的敏感程度对点的敏感排序，存储的内容是点的坐标比如position[0]表示敏感程度最高
        self.adv_trace = None    
        self.tau = None
        self.eta = None


    
    def sensitive_fitness(self, logits, ground_truth):
        # print("sensitive_fitness")
        ground_truth = ground_truth.to(self.device)
        loss = F.cross_entropy(logits, ground_truth, reduction='mean')
        correct = (logits.argmax(-1)==ground_truth.argmax(-1)).sum().item()
        return loss.item(),correct
        
    
    def sensitive_results(self,test_traces,ground_truth,N=1):
        """
        向量化进行敏感性分析,返回每个代表位置插入一个包的对抗序列。
        test_traces: shape = (5000, 5000, 1) 原始测试序列
        ground_truth: shape = (5000,100) 原数据集真实标签集合
        """
        self.eta = np.zeros(test_traces.shape[1])
        self.tau = np.ones(test_traces.shape[1])  # 信息素
        fit_results = []

        test_traces = test_traces.squeeze()
        sample_num, seq_length = test_traces.shape
        dim_num = int(N*seq_length)
        positions_to_test = np.array(sorted(random.sample(range(seq_length), dim_num)))#如果对所有点进行敏感性分析，就改成N=1
        
        
        for pos in positions_to_test:
            perturbed_traces = np.zeros((sample_num, seq_length + 1), dtype=test_traces.dtype)#这里也是要改，如果所有点敏感性分析,第一个维度改为seq_length
            perturbed_traces[:,:pos] = test_traces[:,:pos]
            perturbed_traces[:, pos] = 1  # 注入虚拟包，方向固定为 +1
            perturbed_traces[:, pos+1:] = test_traces[:, pos:]
            perturbed_traces = perturbed_traces[:,:5000]
            
            sum_fitness = 0
            for j in range(0,perturbed_traces.shape[0],batch_size):
                perturbed_traces_tensor = torch.tensor(perturbed_traces[j:j+batch_size]).to(self.device).unsqueeze(-1)
                with torch.no_grad():
                    logits = self.model(perturbed_traces_tensor)
                result = self.sensitive_fitness(logits,ground_truth[j:j+logits.shape[0]].to(self.device))
                sum_fitness = sum_fitness + result[0]
            self.eta[pos] = sum_fitness/perturbed_traces.shape[0]
            print("self.eta[",pos,"]:"," ",self.eta[pos])
            fit_results.append(sum_fitness)
            
        #返回敏感插入位置
        self.eta = (self.eta - self.eta.min()) / (self.eta.max() - self.eta.min() + 1e-8)
        return None
        

    def construct_solution(self):
        solutions = []
        print("tau max:",self.tau.max(),"position max:",self.eta.max())
        for _ in range(self.numant):
            selected_indices = []
            #目前做法是对所有点里面选择
            available = list(range(0, self.original_trace.shape[1]))

            for _ in range(self.max_inject):
                # probs = self.tau[available] ** self.alpha * self.eta[available] ** self.beta
                probs = self.tau[available] ** self.alpha * self.eta[available] ** self.beta
                
                probs = probs / np.sum(probs)
                chosen_idx = np.random.choice(available, p=probs)#选出插入点
                selected_indices.append(chosen_idx)
                # available.remove(chosen_idx)
            selected_positions = [i for i in selected_indices]
            solutions.append(selected_positions)
        return solutions  #构造的解方案，包含插入点，要在哪个位置插入包



    def apply_perturbation(self, solution, pad_values=None):
        """
        矢量化的最快实现
        在3D数组的第二个维度的指定位置插入值
        
        参数:
        - x: 3D数组 (batch_size, seq_len, features)
        - positions: 插入位置的列表
        - pad_values: 插入的值，可以是标量或数组
        
        返回:
        - 在指定位置插入值后的数组（保持原长度）
        """
        N = len(solution)
        
        if pad_values is None:
            pad_values = 1
        if np.isscalar(pad_values):
            pad_values = np.full(N, pad_values)
        
        # 创建扩展后的数组
        extended_length = self.original_trace.shape[1] + N
        extended_x = np.zeros((self.original_trace.shape[0], extended_length, self.original_trace.shape[2]), dtype=self.original_trace.dtype)
        
        # 创建插入标记
        insert_mask = np.zeros(extended_length, dtype=bool)
        
        # 对位置排序
        sorted_indices = np.argsort(solution)
        sorted_positions = solution[sorted_indices]
        sorted_values = pad_values[sorted_indices]
        
        # 调整插入位置
        adjusted_positions = sorted_positions + np.arange(N)
        insert_mask[adjusted_positions] = True
        
        # 填充插入的值
        for i, (pos, val) in enumerate(zip(adjusted_positions, sorted_values)):
            extended_x[:, pos, :] = val
        
        # 填充原始数据
        original_positions = np.where(~insert_mask)[0]
        extended_x[:, original_positions, :] = self.original_trace
        
        # 截断到原长度
        return extended_x[:, :self.original_trace.shape[1], :]

    
    def random_sample_without_replacement(self, num_samples=1000):
        num_total = self.original_trace.shape[0]
        if num_samples > num_total:
            raise ValueError(f"Cannot sample {num_samples} from {num_total} without replacement")
        # 生成不重复的随机索引
        random_indices = np.random.choice(num_total, size=num_samples, replace=False)

        return random_indices

    def run(self, label):
        label = torch.tensor(label)
        
        best_fitness = -float('inf')
        best_correct = -float('inf')
        best_solution = None
        index = self.random_sample_without_replacement()
        #敏感性分析，找出各个点的插入敏感性
        self.sensitive_results(self.original_trace[index],label[index])
        # print(1)
        for iteration in tqdm(range(self.max_iter)):
            solutions = self.construct_solution()
            solutions = np.array(solutions)
            fitness_list = []
            for solution in solutions:
                perturbed_trace = self.apply_perturbation(solution)
                fitness = 0
                sum_correct = 0
                # print("perturbed_trace shape:",perturbed_trace.shape)
                for j in range(0,perturbed_trace.shape[0],batch_size):
                    perturbed_trace_tensor = torch.tensor(perturbed_trace[j:j+batch_size]).to(self.device)
                    with torch.no_grad():
                        logits = self.model(perturbed_traces_tensor)

                    result = self.sensitive_fitness(logits,label[j:j+logits.shape[0]])
                    fitness = fitness + result[0]
                    
                    sum_correct = sum_correct + result[1]
                # print("sum_correct:",sum_correct,"fit:",fitness/64)
                fitness_list.append(fitness)
                # 更新最优解
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution
                    best_correct = sum_correct
            fitness_list = np.array(fitness_list)
    
            # 信息素更新
            self.tau *= (1 - self.rho)
            max_fit = fitness_list.max()
            min_fit = fitness_list.min()
            for i, fit in zip(solutions, fitness_list):
                print("solution:",i)
                for idx in i:
                    self.tau[idx] += (fit-min_fit)/(max_fit-min_fit)
            self.tau = (self.tau - self.tau.min()) / (self.tau.max() - self.tau.min() + 1e-8)
            
            print(f"[Iter {iteration}] Best fitness: {best_fitness:.4f} 攻击成功样本率 {1-(best_correct/self.original_trace.shape[0]):.4f}",self.original_trace.shape[0]-best_correct,"/",self.original_trace.shape[0])
            torch.cuda.empty_cache()     # 释放未使用显存回操作系统
            torch.cuda.ipc_collect()     # 收集 IPC 共享内存

        return best_solution, best_fitness