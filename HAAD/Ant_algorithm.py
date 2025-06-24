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

loss = nn.CrossEntropyLoss()




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
        self.positions = None
        self.adv_trace = None    
        self.tau = None
        self.eta = None

    def sensitive_generate(self, test_traces):
        """
        假定seq_length = 5000
        向量化进行敏感性分析,返回每个代表位置插入一个包的对抗序列。
        test_traces: shape = (5000, 5000) 原始测试序列
        select_p_num: int 选择这么多个敏感位置
        """
        test_traces = test_traces.squeeze()
        batch_size, seq_length = test_traces.shape
        # positions_to_test = np.array(sorted(random.sample(range(seq_length), 100)))#如果对所有点进行敏感性分析，就改成test_traces.shape[1]
        positions_to_test = np.array(range(1,seq_length))#对所有点进行敏感性分析
        

        perturbed_traces = np.zeros((seq_length, batch_size, seq_length + 1), dtype=test_traces.dtype)#这里也是要改，如果所有点敏感性分析,第一个维度改为seq_length

        for i, pos in enumerate(positions_to_test):
            perturbed_traces[i, :, :pos] = test_traces[:, :pos]
            perturbed_traces[i, :, pos] = 1  # 注入虚拟包，方向固定为 +1
            perturbed_traces[i, :, pos+1:] = test_traces[:, pos:]

        return perturbed_traces[:,:,:seq_length], positions_to_test  # 都是array数组，shape = (select_p_num, 样本数, seq_length), 随机选择的敏感点
    
    def sensitive_fitness(self, logits, ground_truth):
        loss = F.cross_entropy(logits, ground_truth, reduction='mean')
        correct = (logits.argmax(-1)==ground_truth.argmax(-1)).sum().item()
        return loss.item(),correct
        
    
    def sensitive_results(self,test_traces,ground_truth):
        """
        向量化进行敏感性分析,返回每个代表位置插入一个包的对抗序列。
        test_traces: shape = (5000, 5000, 1) 原始测试序列
        ground_truth: shape = (5000,100) 原数据集真实标签集合
        """
        # print(ground_truth.shape)
        fit_results = []
        correct_results = []
        perturbed_traces, positions_to_test = self.sensitive_generate(test_traces)
        #转为tensor
        perturbed_traces = torch.tensor(perturbed_traces[:,:,:,np.newaxis]).to(self.device)
        ground_truth = ground_truth.clone().detach()
        
        for i in range(len(positions_to_test)):
            sum_fitness = 0
            sum_correct = 0
            for j in range(0,perturbed_traces.shape[1],100):
                logits = self.model(perturbed_traces[i][j:j+100])
                result = self.sensitive_fitness(logits,ground_truth[j:j+logits.shape[0]])
                sum_fitness = sum_fitness + result[0]
                sum_correct = sum_correct + result[1]
            fit_results.append(sum_fitness)
            correct_results.append(sum_correct)
            
        #储存结果，交叉熵损失结果，预测准确的样本结果
        fit_results = np.array(fit_results)
        correct_results = np.array(correct_results)
        sorted_index = sorted(range(len(fit_results)),key = lambda i:fit_results[i],reverse=True)
        
        #返回敏感插入位置
        self.positions = positions_to_test[sorted_index]   #numpy shape:(num,)
        self.tau = np.ones(test_traces.shape[1])  # 信息素
        self.eta = np.ones(test_traces.shape[1])  # 启发因子（默认敏感度等权）
        return positions_to_test[sorted_index]
        

    def construct_solution(self):
        # pdb.set_trace()
        solutions = []
        for _ in range(self.numant):
            selected_indices = []
            available = list(range(len(self.positions)))
            for _ in range(self.max_inject):
                probs = self.tau[available] ** self.alpha * self.eta[available] ** self.beta
                probs = probs / np.sum(probs)
                chosen_idx = np.random.choice(available, p=probs)
                selected_indices.append(chosen_idx)
                available.remove(chosen_idx)
            selected_positions = [self.positions[i] for i in selected_indices]
            solutions.append(selected_positions)
        return solutions



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
        # pdb.set_trace()
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

    
    def random_sample_without_replacement(self, num_samples=5000):
        num_total = self.original_trace.shape[0]
        if num_samples > num_total:
            raise ValueError(f"Cannot sample {num_samples} from {num_total} without replacement")
        # 生成不重复的随机索引
        random_indices = np.random.choice(num_total, size=num_samples, replace=False)

        return random_indices

    def run(self, label):
        # pdb.set_trace()
        label = torch.tensor(label).to(self.device)
        best_fitness = -float('inf')
        best_correct = -float('inf')
        best_solution = None
        index = self.random_sample_without_replacement()
        #敏感性分析，找出
        self.sensitive_results(self.original_trace[index],label[index])
        
        for iteration in range(self.max_iter):
            solutions = self.construct_solution()
            solutions = np.array(solutions)
            fitness_list = []
            for solution in solutions:
                perturbed_trace = self.apply_perturbation(solution)
                
                
                fitness = 0
                sum_correct = 0
                # print("perturbed_trace shape:",perturbed_trace.shape)
                for j in range(0,perturbed_trace.shape[0],100):
                    perturbed_trace_tensor = torch.tensor(perturbed_trace[j:j+100]).to(self.device)
                    logits = self.model(perturbed_trace_tensor)
                    # print(label[j:j+logits.shape[0]].shape)
                    result = self.sensitive_fitness(logits,label[j:j+logits.shape[0]])
                    fitness = fitness + result[0]
                    
                    sum_correct = sum_correct + result[1]
                print("sum_correct:",sum_correct)
                fitness_list.append(fitness)
                

                # 更新最优解
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution
                    best_correct = sum_correct

            # 信息素更新
            self.tau *= (1 - self.rho)
            for i, fit in zip(solutions, fitness_list):
                for idx in i:
                    self.tau[idx] += fit

            print(f"[Iter {iteration}] Best fitness: {best_fitness:.4f} Defense rate: {best_correct/self.original_trace.shape[0]:.4f}")

        return best_solution, best_fitness