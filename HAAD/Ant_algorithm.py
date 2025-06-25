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
    def __init__(self, model, original_trace, numant, max_inject, max_iter, alpha=1, beta=1, rho=0.1, q=1):
        self.numant = numant
        self.max_inject = max_inject
        self.max_iter = max_iter
        self.original_trace = original_trace.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.alpha = alpha
        self.beta = beta
        #信息素更新步长因子
        self.q = q
        self.rho = rho
        self.positions = None   #按照敏感点的敏感程度对点的敏感排序，存储的内容是点的坐标比如position[0]表示敏感程度最高
        self.adv_trace = None   
        #信息素 
        self.tau = None
        
        #启发因子
        self.eta = None
        
        #多个解向量
        self.solutions = None
        #最优解
        self.best_solution = None
        
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
        positions_to_test = np.array(range(0,seq_length))#对所有点进行敏感性分析
        

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
        print("Sensitivity Analysis....")
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
            fit_results.append(sum_fitness/perturbed_traces.shape[1])
            correct_results.append(sum_correct)
            
        #储存结果，交叉熵损失结果，预测准确的样本结果
        fit_results = np.array(fit_results)
        correct_results = np.array(correct_results)
        sorted_index = sorted(range(len(fit_results)),key = lambda i:fit_results[i],reverse=True)
        
        #返回敏感插入位置
        self.positions = positions_to_test[sorted_index]   #numpy shape:(num,)
        self.tau = np.ones(test_traces.shape[1])  # 信息素
        self.eta = fit_results # 启发因子（默认敏感度等权）
        self.eta = (self.eta - self.eta.min()) / (self.eta.max() - self.eta.min() + 1e-8)
        return positions_to_test[sorted_index]
        

    def construct_solution(self):
        solutions = []
        print("tau max:",self.tau.max(),"position max:",self.eta.max())
        for _ in range(self.numant):
            selected_indices = []
            #目前做法是对所有点里面选择
            available = list(range(0, self.original_trace.shape[1]))
            eta = self.eta.copy()
            for _ in range(self.max_inject):
                # probs = self.tau[available] ** self.alpha * self.eta[available] ** self.beta
                probs = self.tau[available] ** self.alpha * eta[available] ** self.beta
                probs = probs / np.sum(probs)
                chosen_idx = np.random.choice(available, p=probs)#选出插入点
                selected_indices.append(chosen_idx)
                eta[chosen_idx] *= 0.9
            selected_positions = [i for i in selected_indices]
            solutions.append(selected_positions)
            self.solutions = np.array(solutions)
        return solutions  #构造的解方案，包含插入点，要在哪个位置插入包



    def apply_perturbation(self, solution, test_traces=None, pad_values=None):
        """
        矢量化的最快实现
        在3D数组的第二个维度的指定位置插入值
        
        参数:
        
        - solution: 插入位置的列表
        - test_traces: 测试流，如果没有就是原始所有流
        - pad_values: 插入的值，可以是标量或数组
        
        返回:
        - 在指定位置插入值后的数组（保持原长度）
        """
        N = len(solution)
        
        if pad_values is None:
            pad_values = 1
        if np.isscalar(pad_values):
            pad_values = np.full(N, pad_values)
        
        if test_traces is None:
            test_traces = self.original_trace
        
        # 创建扩展后的数组
        extended_length = test_traces.shape[1] + N
        extended_x = np.zeros((test_traces.shape[0], extended_length, test_traces.shape[2]), dtype=self.original_trace.dtype)
        
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
        extended_x[:, original_positions, :] = test_traces
        
        # 截断到原长度
        return extended_x[:, :test_traces.shape[1], :]

    
    def random_sample_without_replacement(self, num_samples=5000):
        num_total = self.original_trace.shape[0]
        if num_samples > num_total:
            raise ValueError(f"Cannot sample {num_samples} from {num_total} without replacement")
        # 生成不重复的随机索引
        random_indices = np.random.choice(num_total, size=num_samples, replace=False)
        # 返回索引
        return random_indices

    
    def update_tau(self,fitness_list, labels):
        """
        信息素更新函数
        - fitness_list: 奖励值存储列表 ndarray shape:[蚂蚁数量，]
        - labels: [num,100]
        无返回值
        """
        index = self.random_sample_without_replacement(1000)
        self.tau *= (1 - self.rho)
        counter = np.zeros_like(self.tau)
        delta = np.zeros_like(self.tau)
        test_traces = self.original_trace[index]
        test_labels = labels[index]
        for i, fit in zip(self.solutions, fitness_list):
            
            unique, counts = np.unique(i, return_counts=True)
            for pos, count in zip(unique, counts):
                temp = np.array([x for x in i if x != pos])
                fit_removed = self.fitness_(self.apply_perturbation(temp, test_traces), test_labels)
                marginal = (fit - fit_removed[0]) / count  # 平均每次的边际效用
                delta[pos] += marginal
                counter[pos] += 1
            
        self.tau += self.q * delta / (counter + 1e-8)
        self.tau = (self.tau - self.tau.min()) / (self.tau.max() - self.tau.min() + 1e-8)
        return None
    
    def fitness_(self,perturbed_traces,labels):
        """
        奖励值反馈函数
        seq_len = 1000
        - perturbed_traces shape:(num,seq_len,1)
        - labels shape:(num,100) one-hot 
        返回值
        - fitness 基于loss的奖励值，越大说明预测越不准
        - sum_correct 预测的准确样本
        """
        fitness = 0
        sum_correct = 0
        
        for j in range(0,perturbed_traces.shape[0],100):
            
            perturbed_trace_tensor = torch.tensor(perturbed_traces[j:j+100]).to(self.device)
            #获取反馈值
            logits = self.model(perturbed_trace_tensor)
            result = self.sensitive_fitness(logits,labels[j:j+logits.shape[0]])

            fitness = fitness + result[0]
            sum_correct = sum_correct + result[1]
        return fitness/perturbed_traces.shape[0], sum_correct

    def run(self, label):
        label = torch.tensor(label).to(self.device)
        
        best_fitness = -float('inf')
        best_correct = -float('inf')
        best_solution = None
        index = self.random_sample_without_replacement()
        #敏感性分析，找出各个点的插入敏感性
        self.sensitive_results(self.original_trace[index],label[index])
        
        for iteration in range(self.max_iter):
            #构造解
            self.construct_solution()     
            fitness_list = []
            for solution in self.solutions:
                #应用扰动
                perturbed_traces = self.apply_perturbation(solution)
                #计算反馈奖励值 perturbed_traces shape:(num, 1000, 1) label shape: (num,100)
                fitness, sum_correct = self.fitness_(perturbed_traces, label)
                print("sum_correct:",sum_correct,"fit:",fitness)
                fitness_list.append(fitness)
                # 更新最优解
                if fitness > best_fitness:
                    best_fitness = fitness
                    self.best_solution = solution
                    best_correct = sum_correct
            fitness_list = np.array(fitness_list)
            
            # 信息素更新
            self.update_tau(fitness_list,label)       
            # print result
            print(f"[Iter {iteration}] Best fitness: {best_fitness:.4f} Defense rate: {1-best_correct/self.original_trace.shape[0]:.4f}")

        return self.best_solution, best_fitness