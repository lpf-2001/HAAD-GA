# blackbox_universal_haad.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import gc


class UniversalHAAD:
    """
    通用扰动黑盒蚁群算法
    - 纯黑盒优化，无梯度信息
    - 生成通用扰动模式，应用于所有样本
    - 基于模型输出的适应度评估
    - 显存友好的实现
    """

    def __init__(
        self,
        model: nn.Module,
        original_trace: torch.Tensor,
        numant: int = 50,
        max_inject: int = 10,
        max_iter: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
        rho: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        inject_value: Union[float, str] = 1.0,
        perturb_mode: str = "overwrite",
        use_amp: bool = True,
        memory_limit_mb: float = 8192,
        eval_chunk_size: int = 512,
    ):
        """
        Args:
            model: 目标模型
            original_trace: 原始数据 [B, L] 或 [B, L, C]
            numant: 蚂蚁数量
            max_inject: 每个解的最大扰动位置数
            max_iter: 最大迭代数
            alpha: 信息素重要性参数
            beta: 启发式信息重要性参数  
            rho: 信息素蒸发率
            inject_value: 扰动值，可以是具体数值或'random'
            perturb_mode: 扰动模式 'overwrite' 或 'insert'
            use_amp: 是否使用混合精度
            memory_limit_mb: 显存限制(MB)
            eval_chunk_size: 评估时的批量大小
        """
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
        self.eta = torch.ones(self.L, device=self.device, dtype=dtype)  # 初始化为均匀分布

        # 状态变量
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # 模型配置
        self.model.to(self.device)
        self.model.eval()

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
            return torch.rand(shape, device=self.device, dtype=self.original_trace.dtype)
        else:
            return torch.full(shape, float(self.inject_value), 
                            device=self.device, dtype=self.original_trace.dtype)

    @torch.inference_mode()
    def apply_universal_perturbation(
        self, 
        data: torch.Tensor, 
        perturbation_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        应用通用扰动到数据
        
        Args:
            data: 输入数据 [B, L, C]
            perturbation_positions: 扰动位置 [K] (K <= max_inject)
            
        Returns:
            扰动后的数据 [B, L, C]
        """
        if len(perturbation_positions) == 0:
            return data
            
        perturbed = data.clone()
        positions = perturbation_positions.to(self.device)
        
        if self.perturb_mode == "overwrite":
            # 直接覆写指定位置
            inject_vals = self._generate_inject_value((data.size(0), len(positions), data.size(2)))
            perturbed[:, positions, :] = inject_vals
            
        elif self.perturb_mode == "insert":
            # 在指定位置插入扰动（需要截断以保持长度）
            B, L, C = data.shape
            K = len(positions)
            
            # 创建扩展数据
            extended = torch.zeros(B, L + K, C, device=self.device, dtype=data.dtype)
            
            # 排序位置以正确插入
            sorted_pos, sort_idx = torch.sort(positions)
            
            current_orig = 0
            current_ext = 0
            
            for i, pos in enumerate(sorted_pos):
                # 复制原始数据到插入位置之前
                if pos > current_orig:
                    copy_len = pos - current_orig
                    extended[:, current_ext:current_ext+copy_len, :] = data[:, current_orig:pos, :]
                    current_ext += copy_len
                    current_orig = pos
                
                # 插入扰动值
                inject_val = self._generate_inject_value((B, 1, C))
                extended[:, current_ext, :] = inject_val.squeeze(1)
                current_ext += 1
            
            # 复制剩余的原始数据
            if current_orig < L:
                extended[:, current_ext:current_ext+(L-current_orig), :] = data[:, current_orig:, :]
            
            # 截断到原始长度
            perturbed = extended[:, :L, :]
        
        return perturbed

    @torch.inference_mode()
    def evaluate_fitness(
        self, 
        perturbation_positions: torch.Tensor, 
        labels: torch.Tensor,
        fitness_type: str = "loss_increase"
    ) -> float:
        """
        评估扰动方案的适应度（纯黑盒）
        
        Args:
            perturbation_positions: 扰动位置
            labels: 真实标签
            fitness_type: 适应度类型
                - "loss_increase": 损失增加量
                - "accuracy_drop": 准确率下降
                - "confidence_drop": 置信度下降
                
        Returns:
            适应度值（越大越好）
        """
        if len(perturbation_positions) == 0:
            return 0.0
            
        total_fitness = 0.0
        total_samples = 0
        
        # 分块处理以节省显存
        for start_idx in range(0, self.B, self.eval_chunk_size):
            end_idx = min(start_idx + self.eval_chunk_size, self.B)
            
            # 获取数据块
            data_chunk = self.original_trace[start_idx:end_idx]
            label_chunk = labels[start_idx:end_idx]
            target_chunk = self._to_class_indices(label_chunk)
            
            # 计算原始输出
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                original_logits = self.model(data_chunk)
                original_loss = F.cross_entropy(original_logits, target_chunk, reduction='mean')
                
            # 应用扰动
            perturbed_chunk = self.apply_universal_perturbation(data_chunk, perturbation_positions)
            
            # 计算扰动后输出
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                perturbed_logits = self.model(perturbed_chunk)
                perturbed_loss = F.cross_entropy(perturbed_logits, target_chunk, reduction='mean')
            
            # 计算适应度
            chunk_size = end_idx - start_idx
            
            if fitness_type == "loss_increase":
                # 损失增加量
                fitness_chunk = float(perturbed_loss - original_loss) * chunk_size
                
            elif fitness_type == "accuracy_drop":
                # 准确率下降
                orig_correct = (original_logits.argmax(dim=-1) == target_chunk).sum().float()
                pert_correct = (perturbed_logits.argmax(dim=-1) == target_chunk).sum().float()
                fitness_chunk = float(orig_correct - pert_correct)
                
            elif fitness_type == "confidence_drop":
                # 目标类别置信度下降
                orig_conf = F.softmax(original_logits, dim=-1)[torch.arange(chunk_size), target_chunk]
                pert_conf = F.softmax(perturbed_logits, dim=-1)[torch.arange(chunk_size), target_chunk]
                fitness_chunk = float((orig_conf - pert_conf).sum())
                
            else:
                raise ValueError(f"未知的适应度类型: {fitness_type}")
            
            total_fitness += fitness_chunk
            total_samples += chunk_size
            
            # 清理中间变量
            del data_chunk, label_chunk, target_chunk
            del original_logits, perturbed_logits, original_loss, perturbed_loss, perturbed_chunk
        
        # 清理显存
        self._cleanup_gpu_memory()
        
        return total_fitness / max(total_samples, 1)

    @torch.inference_mode()
    def update_heuristic_info(
        self, 
        labels: torch.Tensor, 
        sample_size: Optional[int] = None,
        fitness_type: str = "loss_increase"
    ):
        """
        通过单位置扰动更新启发式信息（黑盒方式）
        
        Args:
            labels: 标签
            sample_size: 采样数量，None表示使用全部数据
            fitness_type: 适应度类型
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
        
        # 逐位置计算启发式信息
        for pos in tqdm(range(self.L), desc="计算位置敏感性", leave=False):
            position_tensor = torch.tensor([pos], device=self.device)
            
            # 计算单个位置的适应度
            fitness = 0.0
            total_samples = 0
            
            for start_idx in range(0, sample_data.size(0), self.eval_chunk_size):
                end_idx = min(start_idx + self.eval_chunk_size, sample_data.size(0))
                
                data_chunk = sample_data[start_idx:end_idx]
                label_chunk = sample_labels[start_idx:end_idx]
                target_chunk = self._to_class_indices(label_chunk)
                
                # 原始预测
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    orig_logits = self.model(data_chunk)
                    
                # 扰动后预测
                pert_chunk = self.apply_universal_perturbation(data_chunk, position_tensor)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pert_logits = self.model(pert_chunk)
                
                # 计算适应度贡献
                chunk_size = end_idx - start_idx
                
                if fitness_type == "loss_increase":
                    orig_loss = F.cross_entropy(orig_logits, target_chunk, reduction='mean')
                    pert_loss = F.cross_entropy(pert_logits, target_chunk, reduction='mean')
                    fitness += float(pert_loss - orig_loss) * chunk_size
                    
                elif fitness_type == "accuracy_drop":
                    orig_correct = (orig_logits.argmax(dim=-1) == target_chunk).sum().float()
                    pert_correct = (pert_logits.argmax(dim=-1) == target_chunk).sum().float()
                    fitness += float(orig_correct - pert_correct)
                    
                total_samples += chunk_size
                
                del data_chunk, label_chunk, target_chunk, orig_logits, pert_logits, pert_chunk
            
            eta_values[pos] = fitness / max(total_samples, 1)
        
        # 归一化启发式信息
        eta_min, eta_max = eta_values.min(), eta_values.max()
        if eta_max - eta_min > 1e-8:
            self.eta = (eta_values - eta_min) / (eta_max - eta_min + 1e-8)
        else:
            self.eta = torch.ones_like(eta_values) / self.L
            
        # 确保所有值为正
        self.eta = self.eta.clamp_min(1e-8)
        
        self._cleanup_gpu_memory()

    def construct_solutions(self) -> List[torch.Tensor]:
        """基于信息素和启发式信息构造解"""
        # 计算选择概率

        weights1 = (self.tau ** self.alpha) 
        weights2 = (self.eta ** self.beta)
        weights = weights1 * weights2
        probabilities = weights / weights.sum()
        
        solutions = []
        for _ in tqdm(range(self.numant),desc="蚂蚁数",leave=False):
            # 使用轮盘赌选择，无重复
            if self.max_inject <= self.L:
                chosen = torch.multinomial(probabilities, self.max_inject, replacement=False)
            else:
                chosen = torch.multinomial(probabilities, self.max_inject, replacement=True)
            solutions.append(chosen)
            
        return solutions

    def update_pheromone(
        self, 
        solutions: List[torch.Tensor], 
        fitness_values: List[float],
        strategy: str = "elitist"
    ):
        """
        信息素更新
        
        Args:
            solutions: 解的列表
            fitness_values: 对应的适应度值
            strategy: 更新策略 ("elitist", "rank", "proportional")
        """
        # 信息素蒸发
        self.tau *= (1.0 - self.rho)
        
        fitness_tensor = torch.tensor(fitness_values, device=self.device)
        
        if strategy == "elitist":
            # 只有最优解更新信息素
            best_idx = fitness_tensor.argmax()
            best_solution = solutions[best_idx]
            best_fitness = fitness_values[best_idx]
            
            # 归一化适应度值作为更新量
            if len(fitness_values) > 1:
                fitness_range = max(fitness_values) - min(fitness_values)
                update_amount = best_fitness / max(fitness_range, 1e-8)
            else:
                update_amount = 1.0
                
            self.tau[best_solution] += update_amount
            
        elif strategy == "rank":
            # 基于排名的更新
            sorted_indices = fitness_tensor.argsort(descending=True)
            for rank, idx in enumerate(sorted_indices):
                weight = (len(solutions) - rank) / len(solutions)
                self.tau[solutions[idx]] += weight
                
        elif strategy == "proportional":
            # 比例更新
            if fitness_tensor.max() > fitness_tensor.min():
                normalized_fitness = (fitness_tensor - fitness_tensor.min()) / (fitness_tensor.max() - fitness_tensor.min())
                for sol, weight in zip(solutions, normalized_fitness):
                    self.tau[sol] += float(weight)
        
        # 归一化信息素，避免数值问题
        tau_min, tau_max = self.tau.min(), self.tau.max()
        if tau_max - tau_min > 1e-8:
            self.tau = (self.tau - tau_min) / (tau_max - tau_min + 1e-8)
        
        # 确保信息素在合理范围内
        self.tau = self.tau.clamp(1e-8, 10.0)

    @torch.inference_mode()
    def run(
        self,
        labels: torch.Tensor,
        heuristic_sample_size: Optional[int] = 1000,
        fitness_type: str = "loss_increase",
        pheromone_strategy: str = "elitist",
        verbose: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        运行通用扰动优化
        
        Args:
            labels: 目标标签
            heuristic_sample_size: 启发式信息计算的采样大小
            fitness_type: 适应度函数类型
            pheromone_strategy: 信息素更新策略
            verbose: 是否显示详细信息
            
        Returns:
            (最优扰动位置, 最优适应度)
        """
        labels = torch.as_tensor(labels, device=self.device)
        
        if verbose:
            print(f"数据形状: {self.original_trace.shape}")
            print(f"蚂蚁数量: {self.numant}, 最大扰动位置: {self.max_inject}")
            print(f"适应度类型: {fitness_type}")
        
        # 初始化启发式信息
        self.update_heuristic_info(
            labels, 
            sample_size=heuristic_sample_size,
            fitness_type=fitness_type
        )
        
        # 蚁群优化主循环
        best_fitness = -float('inf')
        best_solution = None
        
        for iteration in tqdm(range(self.max_iter), desc="ACO优化"):
            # 构造解
            solutions = self.construct_solutions()
            
            # 评估所有解
            fitness_values = []
            for solution in solutions:
                fitness = self.evaluate_fitness(solution, labels, fitness_type)
                fitness_values.append(fitness)
                
                # 更新全局最优
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.clone()
            
            # 更新信息素
            self.update_pheromone(solutions, fitness_values, pheromone_strategy)
            
            # 记录历史
            self.fitness_history.append(max(fitness_values))
            
            if verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                current_best = max(fitness_values)
                avg_fitness = sum(fitness_values) / len(fitness_values)
                print(f"[Iter {iteration+1}] 最优适应度: {best_fitness:.6f}, "
                      f"当前轮最优: {current_best:.6f}, 平均: {avg_fitness:.6f}")
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        if verbose:
            print(f"\n优化完成!")
            print(f"最优扰动位置: {best_solution.cpu().numpy()}")
            print(f"最优适应度: {best_fitness:.6f}")
        
        return best_solution, best_fitness

    def get_universal_perturbation_info(self) -> dict:
        """获取通用扰动的详细信息"""
        if self.best_solution is None:
            return {"error": "尚未运行优化"}
        
        return {
            "perturbation_positions": self.best_solution.cpu().numpy().tolist(),
            "num_positions": len(self.best_solution),
            "fitness": self.best_fitness,
            "perturbation_ratio": len(self.best_solution) / self.L,
            "fitness_history": self.fitness_history,
            # "final_tau": self.tau.cpu().numpy(),
            # "final_eta": self.eta.cpu().numpy(),
        }

    def apply_to_new_data(self, new_data: torch.Tensor) -> torch.Tensor:
        """将找到的通用扰动应用到新数据"""
        if self.best_solution is None:
            raise ValueError("尚未找到最优解，请先运行 run() 方法")
            
        new_data = self._preprocess_data(new_data, self.original_trace.dtype)
        return self.apply_universal_perturbation(new_data, self.best_solution)