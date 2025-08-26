import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import gc
import numpy as np


class UniversalHAAD:
    """
    修复版本的通用扰动黑盒蚁群算法
    主要修复：
    1. 启发式信息计算的采样率问题
    2. 适应度计算的稳定性
    3. 信息素更新策略
    4. 数值稳定性改进
    """

    def __init__(
        self,
        model: nn.Module,
        original_trace: torch.Tensor,
        numant: int = 50,
        max_inject: int = 10,
        max_iter: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,  # 增加启发式信息权重
        rho: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        inject_value: Union[float, str] = 1.0,
        perturb_mode: str = "overwrite",
        use_amp: bool = True,
        memory_limit_mb: float = 8192,
        eval_chunk_size: int = 512,
    ):
        assert perturb_mode in ("overwrite", "insert")
        
        self.model = model
        self.numant = int(numant)
        self.max_inject = int(max_inject)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.inject_value = inject_value
        self.perturb_mode = perturb_mode
        self.memory_limit_mb = memory_limit_mb
        self.eval_chunk_size = eval_chunk_size

        # 设备配置
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.use_amp = use_amp and (device.type == "cuda")

        # 数据预处理
        self.original_trace = self._preprocess_data(original_trace, dtype)
        self.B, self.L, self.C = self.original_trace.shape

        # 初始化信息素和启发式信息
        self.tau = torch.ones(self.L, device=self.device, dtype=dtype)
        self.eta = torch.ones(self.L, device=self.device, dtype=dtype)

        # 状态变量
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # 模型配置
        self.model.to(self.device)
        self.model.eval()
        
        # 添加适应度基线，用于稳定计算
        self.fitness_baseline = 0.0

    def _preprocess_data(self, data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """预处理数据，统一格式为 [B, L, C]"""
        x = torch.as_tensor(data, dtype=dtype)
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        elif x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # [L] -> [1, L, 1]
        
        assert x.ndim == 3, f"数据维度应为2或3，得到{x.ndim}"
        return x.to(self.device, non_blocking=True).contiguous()

    @staticmethod
    def _to_class_indices(y: torch.Tensor) -> torch.Tensor:
        """将标签转换为类别索引"""
        if y.ndim >= 2 and y.size(-1) > 1:
            return y.argmax(dim=-1).long()
        return y.long().view(-1)

    def _cleanup_gpu_memory(self):
        """清理GPU显存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def _generate_inject_value(self, shape: torch.Size) -> torch.Tensor:
        """生成注入值"""
        if self.inject_value == 'random':
            # 使用更激进的随机值范围
            return torch.randn(shape, device=self.device, dtype=self.original_trace.dtype) * 0.5
        else:
            return torch.full(shape, float(self.inject_value), 
                            device=self.device, dtype=self.original_trace.dtype)

    @torch.inference_mode()
    def apply_universal_perturbation(
        self, 
        data: torch.Tensor, 
        perturbation_positions: torch.Tensor
    ) -> torch.Tensor:
        """应用通用扰动到数据"""
        if len(perturbation_positions) == 0:
            return data
            
        perturbed = data.clone()
        positions = perturbation_positions.to(self.device)
        
        if self.perturb_mode == "overwrite":
            # 直接覆写指定位置
            inject_vals = self._generate_inject_value((data.size(0), len(positions), data.size(2)))
            perturbed[:, positions, :] = inject_vals
            
        elif self.perturb_mode == "insert":
            # 插入模式的实现（保持原有逻辑）
            B, L, C = data.shape
            K = len(positions)
            
            extended = torch.zeros(B, L + K, C, device=self.device, dtype=data.dtype)
            sorted_pos, sort_idx = torch.sort(positions)
            
            current_orig = 0
            current_ext = 0
            
            for i, pos in enumerate(sorted_pos):
                if pos > current_orig:
                    copy_len = pos - current_orig
                    extended[:, current_ext:current_ext+copy_len, :] = data[:, current_orig:pos, :]
                    current_ext += copy_len
                    current_orig = pos
                
                inject_val = self._generate_inject_value((B, 1, C))
                extended[:, current_ext, :] = inject_val.squeeze(1)
                current_ext += 1
            
            if current_orig < L:
                extended[:, current_ext:current_ext+(L-current_orig), :] = data[:, current_orig:, :]
            
            perturbed = extended[:, :L, :]
        
        return perturbed

    @torch.inference_mode()
    def evaluate_fitness(
        self, 
        perturbation_positions: torch.Tensor, 
        labels: torch.Tensor,
        fitness_type: str = "accuracy_drop",
        sample_trace: Optional[torch.Tensor] = None
    ) -> float:
        """评估扰动方案的适应度（改进版）"""
        if len(perturbation_positions) == 0:
            return 0.0
            
        total_fitness = 0.0
        total_samples = 0
        if sample_trace is None:
            sample_trace = self.original_trace
        # 分块处理以节省显存
        for start_idx in range(0, sample_trace.shape[0], self.eval_chunk_size):
            end_idx = min(start_idx + self.eval_chunk_size, self.B)
            
            # 获取数据块
            data_chunk = sample_trace[start_idx:end_idx]
            label_chunk = labels[start_idx:end_idx]
            target_chunk = self._to_class_indices(label_chunk)
            chunk_size = end_idx - start_idx
            
            try:
                # 计算原始输出
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    original_logits = self.model(data_chunk)
                    
                # 应用扰动
                perturbed_chunk = self.apply_universal_perturbation(data_chunk, perturbation_positions)
                
                # 计算扰动后输出
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    perturbed_logits = self.model(perturbed_chunk)
                
                # 计算适应度
                if fitness_type == "loss_increase":
                    # 改进的损失计算，使用更稳定的方式
                    original_loss = F.cross_entropy(original_logits, target_chunk, reduction='none')
                    perturbed_loss = F.cross_entropy(perturbed_logits, target_chunk, reduction='none')
                    
                    # 计算每个样本的损失增加，然后求和
                    loss_diff = perturbed_loss - original_loss
                    fitness_chunk = float(loss_diff.sum())
                    
                elif fitness_type == "accuracy_drop":
                    # 准确率下降
                    orig_correct = (original_logits.argmax(dim=-1) == target_chunk).float()
                    pert_correct = (perturbed_logits.argmax(dim=-1) == target_chunk).float()
                    fitness_chunk = float((orig_correct - pert_correct).sum())
                    
                elif fitness_type == "confidence_drop":
                    # 目标类别置信度下降
                    orig_probs = F.softmax(original_logits, dim=-1)
                    pert_probs = F.softmax(perturbed_logits, dim=-1)
                    
                    orig_conf = orig_probs[torch.arange(chunk_size), target_chunk]
                    pert_conf = pert_probs[torch.arange(chunk_size), target_chunk]
                    fitness_chunk = float((orig_conf - pert_conf).sum())
                    
                else:
                    raise ValueError(f"未知的适应度类型: {fitness_type}")
                
                total_fitness += fitness_chunk
                total_samples += chunk_size
                
            except Exception as e:
                print(f"适应度计算出错: {e}")
                # 出错时返回一个小的负值
                total_fitness += -0.1 * chunk_size
                total_samples += chunk_size
            
            # 清理中间变量
            del data_chunk, label_chunk, target_chunk
            if 'original_logits' in locals():
                del original_logits, perturbed_logits, perturbed_chunk
        
        # 清理显存
        self._cleanup_gpu_memory()
        
        # 返回平均适应度
        avg_fitness = total_fitness / max(total_samples, 1)
        return avg_fitness

    @torch.inference_mode()
    def update_heuristic_info(
        self, 
        labels: torch.Tensor, 
        sample_size: Optional[int] = None,
        fitness_type: str = "loss_increase",
        sample_positions: int = 5000  # 限制采样的位置数量
    ):
        """
        更新启发式信息（修复版本）
        """
        print("更新启发式信息...")
        
        # 采样数据以加速计算
        if sample_size is not None and sample_size < self.B:
            sample_idx = torch.randperm(self.B)[:sample_size]
            sample_data = self.original_trace[sample_idx]
            sample_labels = labels[sample_idx]
        else:
            sample_data = self.original_trace
            sample_labels = labels
            
        eta_values = torch.zeros(self.L, device=self.device)
        
        # 采样位置以加速计算，而不是计算所有位置
        if sample_positions < self.L:
            position_indices = torch.randperm(self.L)[:sample_positions].sort()[0]
        else:
            position_indices = torch.arange(self.L)
        
        print(f"计算 {len(position_indices)} 个位置的敏感性（总共{self.L}个位置）")
        
        # 逐位置计算启发式信息
        for i, pos in enumerate(tqdm(position_indices, desc="计算位置敏感性", leave=False)):
            pos = int(pos)
            position_tensor = torch.tensor([pos], device=self.device)
            
            try:
                # 计算单个位置的适应度
                fitness = self.evaluate_fitness(position_tensor, sample_labels, fitness_type, sample_trace=sample_data)
                eta_values[pos] = max(fitness, 0.001)  # 确保最小值为正
            except Exception as e:
                print(f"位置 {pos} 计算出错: {e}")
                eta_values[pos] = 0.001
        
        # 对于未采样的位置，使用平均值
        if sample_positions < self.L:
            sampled_mean = eta_values[position_indices].mean()
            eta_values[eta_values == 0] = sampled_mean
        
        # 改进的归一化启发式信息
        eta_min, eta_max = eta_values.min(), eta_values.max()
        if eta_max > eta_min + 1e-8:
            # 使用更稳定的归一化
            self.eta = ((eta_values - eta_min) / (eta_max - eta_min)) + 0.01
        else:
            # 如果所有值相似，使用均匀分布加小扰动
            self.eta = torch.ones_like(eta_values) + torch.randn_like(eta_values) * 0.01
            
        # 确保所有值为正且在合理范围内
        self.eta = self.eta.clamp_min(0.01).clamp_max(10.0)
        
        print(f"启发式信息更新完成，范围: [{self.eta.min():.4f}, {self.eta.max():.4f}]")
        self._cleanup_gpu_memory()

    def construct_solutions(self) -> List[torch.Tensor]:
        """基于信息素和启发式信息构造解（改进版）"""
        # 计算选择概率，添加数值稳定性
        tau_weighted = torch.pow(self.tau + 1e-8, self.alpha)
        eta_weighted = torch.pow(self.eta + 1e-8, self.beta)
        weights = tau_weighted * eta_weighted
        
        # 添加少量随机噪声以避免过早收敛
        noise = torch.randn_like(weights) * 0.01
        weights = weights + noise.abs()
        
        # 归一化概率
        probabilities = weights / (weights.sum() + 1e-8)
        
        solutions = []
        for ant_id in range(self.numant):
            try:
                # 使用轮盘赌选择
                if self.max_inject <= self.L:
                    chosen = torch.multinomial(probabilities, self.max_inject, replacement=False)
                else:
                    chosen = torch.multinomial(probabilities, self.max_inject, replacement=True)
                solutions.append(chosen)
            except Exception as e:
                print(f"蚂蚁 {ant_id} 构造解时出错: {e}")
                # 回退到随机选择
                chosen = torch.randperm(self.L)[:self.max_inject]
                solutions.append(chosen)
            
        return solutions

    def update_pheromone(
        self, 
        solutions: List[torch.Tensor], 
        fitness_values: List[float],
        strategy: str = "elitist"
    ):
        """信息素更新（改进版）"""
        # 信息素蒸发
        self.tau *= (1.0 - self.rho)
        
        fitness_tensor = torch.tensor(fitness_values, device=self.device)
        
        # 处理适应度值，确保为正值或合理范围
        if fitness_tensor.min() < 0:
            # 将适应度值平移到正数范围
            fitness_tensor = fitness_tensor - fitness_tensor.min() + 0.1
        
        if strategy == "elitist":
            # 精英策略：最优解和前几名都更新信息素
            sorted_indices = fitness_tensor.argsort(descending=True)
            top_k = min(3, len(solutions))  # 取前3名
            
            for rank, idx in enumerate(sorted_indices[:top_k]):
                weight = (top_k - rank) / top_k * fitness_tensor[idx] / (fitness_tensor.max() + 1e-8)
                self.tau[solutions[idx]] += float(weight)
                
        elif strategy == "rank":
            # 基于排名的更新
            sorted_indices = fitness_tensor.argsort(descending=True)
            for rank, idx in enumerate(sorted_indices):
                weight = (len(solutions) - rank) / len(solutions)
                self.tau[solutions[idx]] += weight
                
        elif strategy == "proportional":
            # 比例更新
            if fitness_tensor.max() > fitness_tensor.min():
                normalized_fitness = fitness_tensor / (fitness_tensor.sum() + 1e-8)
                for sol, weight in zip(solutions, normalized_fitness):
                    self.tau[sol] += float(weight)
        
        # 改进的信息素边界控制
        self.tau = self.tau.clamp(0.01, 5.0)  # 限制在更小的范围内
        
        # 添加信息素重新初始化机制，防止过早收敛
        if self.tau.std() < 0.1:  # 如果信息素分布过于均匀
            print("信息素分布过于均匀，添加扰动")
            noise = torch.randn_like(self.tau) * 0.1
            self.tau += noise.abs()
            self.tau = self.tau.clamp(0.01, 5.0)

    @torch.inference_mode()
    def run(
        self,
        labels: torch.Tensor,
        heuristic_sample_size: Optional[int] = 1000,
        fitness_type: str = "loss_increase",
        pheromone_strategy: str = "elitist",
        verbose: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """运行通用扰动优化（改进版）"""
        labels = torch.as_tensor(labels, device=self.device)
        
        if verbose:
            print(f"数据形状: {self.original_trace.shape}")
            print(f"蚂蚁数量: {self.numant}, 最大扰动位置: {self.max_inject}")
            print(f"适应度类型: {fitness_type}")
        
        # 计算适应度基线（无扰动时的适应度）
        try:
            self.fitness_baseline = self.evaluate_fitness(torch.tensor([], device=self.device), labels, fitness_type)
            print(f"适应度基线: {self.fitness_baseline:.6f}")
        except:
            self.fitness_baseline = 0.0
        
        # 初始化启发式信息
        self.update_heuristic_info(
            labels, 
            sample_size=heuristic_sample_size,
            fitness_type=fitness_type,
            sample_positions=min(5000, self.L)  # 限制采样位置数
        )
        
        # 蚁群优化主循环
        best_fitness = -float('inf')
        best_solution = None
        stagnation_count = 0
        
        for iteration in tqdm(range(self.max_iter), desc="ACO优化"):
            # 构造解
            solutions = self.construct_solutions()
            
            # 评估所有解
            fitness_values = []
            valid_solutions = []
            valid_fitness = []
            
            for i, solution in enumerate(solutions):
                try:
                    fitness = self.evaluate_fitness(solution, labels, fitness_type)
                    fitness_values.append(fitness)
                    valid_solutions.append(solution)
                    valid_fitness.append(fitness)
                    
                    if verbose and i % 10 == 0:  # 减少输出频率
                        print(f"Iter {iteration}, Ant {i}: 适应度: {fitness:.6f}")
                    
                    # 更新全局最优
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = solution.clone()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                        
                except Exception as e:
                    print(f"蚂蚁 {i} 评估失败: {e}")
                    # 给失败的解一个很小的适应度值
                    fitness_values.append(-1.0)
                    valid_solutions.append(solution)
                    valid_fitness.append(-1.0)
            
            if not valid_fitness:
                print(f"第 {iteration} 轮所有解都失败，跳过")
                continue
            
            # 更新信息素（只使用有效的解）
            if len(valid_solutions) > 0:
                self.update_pheromone(valid_solutions, valid_fitness, pheromone_strategy)
            
            # 记录历史
            current_best = max(valid_fitness) if valid_fitness else -1.0
            self.fitness_history.append(current_best)
            
            if verbose and (iteration + 1) % max(1, self.max_iter // 20) == 0:
                avg_fitness = sum(valid_fitness) / len(valid_fitness) if valid_fitness else 0
                print(f"[Iter {iteration+1}] 全局最优: {best_fitness:.6f}, "
                      f"当前轮最优: {current_best:.6f}, 平均: {avg_fitness:.6f}")
                print(f"信息素范围: [{self.tau.min():.4f}, {self.tau.max():.4f}]")
            
            # 早停机制
            if stagnation_count > 20 and iteration > self.max_iter // 3:
                print(f"连续 {stagnation_count} 轮无改进，提前停止")
                break
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        if verbose:
            print(f"\n优化完成!")
            if best_solution is not None:
                print(f"最优扰动位置: {best_solution.cpu().numpy()}")
                print(f"最优适应度: {best_fitness:.6f}")
                print(f"相对基线提升: {best_fitness - self.fitness_baseline:.6f}")
            else:
                print("未找到有效解")
        
        return best_solution, best_fitness

    def get_universal_perturbation_info(self) -> dict:
        """获取通用扰动的详细信息"""
        if self.best_solution is None:
            return {"error": "尚未运行优化"}
        
        return {
            "perturbation_positions": self.best_solution.cpu().numpy().tolist(),
            "num_positions": len(self.best_solution),
            "fitness": self.best_fitness,
            "fitness_baseline": self.fitness_baseline,
            "fitness_improvement": self.best_fitness - self.fitness_baseline,
            "perturbation_ratio": len(self.best_solution) / self.L,
            "fitness_history": self.fitness_history,
        }

    def apply_to_new_data(self, new_data: torch.Tensor) -> torch.Tensor:
        """将找到的通用扰动应用到新数据"""
        if self.best_solution is None:
            raise ValueError("尚未找到最优解，请先运行 run() 方法")
            
        new_data = self._preprocess_data(new_data, self.original_trace.dtype)
        return self.apply_universal_perturbation(new_data, self.best_solution)