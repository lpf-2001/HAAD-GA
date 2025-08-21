import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class HAAD:
    """
    显存友好版 HAAD（无梯度、分块、小步前向、不展开巨型批）
    - original_trace: (B, L, 1) 或 (B, L) -> 统一为 (B, L, 1)
    - labels: 在 run(...) 里传入；支持 one-hot 或类别索引
    - 重要参数：
        numant: 蚁群数量
        max_inject: 每只蚂蚁插入点数
        max_iter: 迭代轮数
        alpha, beta, rho: 信息素超参
    - 重要开关：
        perturb_mode: 'overwrite'（默认，最快）或 'insert_truncate'（与原逻辑一致）
        use_amp: True/False（CUDA 下建议 True 省显存）
    """

    def __init__(
        self,
        model: nn.Module,
        original_trace,
        numant: int,
        max_inject: int,
        max_iter: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        rho: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        inject_value: float = 1.0,
        perturb_mode: str = "overwrite",  # 'overwrite' | 'insert_truncate'
        use_amp: bool = True,
    ):
        assert perturb_mode in ("overwrite", "insert_truncate")
        self.model = model
        self.numant = int(numant)
        self.max_inject = int(max_inject)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.inject_value = float(inject_value)
        self.perturb_mode = perturb_mode

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.use_amp = use_amp and (device.type == "cuda")

        # 统一原始数据到 [B, L, 1]
        x = torch.as_tensor(original_trace, dtype=dtype)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        assert x.ndim == 3, "original_trace 应为 (B,L,1) 或 (B,L)"
        self.original_trace = x.to(self.device, non_blocking=True).contiguous()
        self.B, self.L, self.C = self.original_trace.shape

        # 信息素（tau）与启发式信息（eta）
        self.tau = torch.ones(self.L, device=self.device, dtype=dtype)
        self.eta = torch.zeros(self.L, device=self.device, dtype=dtype)

        self.best_solution = None
        self.labels = None  # 在 run() 里赋值

        self.model.to(self.device)
        self.model.eval()

    # ---------- 工具 ----------

    @staticmethod
    def _to_class_indices(y: torch.Tensor) -> torch.Tensor:
        """兼容 one-hot 或 已是索引的标签，返回 (N,) 的 long 索引。"""
        if y.ndim >= 2 and y.size(-1) > 1:
            return y.argmax(dim=-1).long()
        return y.long().view(-1)

    # ---------- 敏感性分析（小步前向，不展开 P*b） ----------

    @torch.inference_mode()
    def sensitive_results(
        self,
        sample_indices: Optional[torch.Tensor],
        labels: torch.Tensor,
        sample_chunk: int = 512,
        pos_chunk: int = 64,
    ):
        """
        计算每个位置的启发式 eta。
        绝不构造 [P*b, L, 1]；而是对每个位置单点扰动，小批量前向，显存恒定。
        """
        B, L, _ = self.original_trace.shape
        device = self.device

        if sample_indices is None:
            k = min(1000, B)
            sample_indices = torch.randperm(B, device=device)[:k]
        else:
            sample_indices = torch.as_tensor(sample_indices, device=device, dtype=torch.long)

        x_all = self.original_trace.index_select(0, sample_indices)  # [Bsub, L, 1]
        y_all = torch.as_tensor(labels, device=device)
        y_all = y_all.index_select(0, sample_indices)                # [Bsub, ...]
        Bsub = x_all.size(0)

        eta_sum = torch.zeros(L, device=device, dtype=x_all.dtype)

        for s in range(0, Bsub, sample_chunk):
            e = min(s + sample_chunk, Bsub)
            x_chunk = x_all[s:e].clone()         # [b, L, 1]
            y_chunk = y_all[s:e]                 # [b, ...]

            t_chunk = self._to_class_indices(y_chunk)

            for p0 in range(0, L, pos_chunk):
                p1 = min(p0 + pos_chunk, L)
                # 对这个区间内的每个位置分别前向（每次 batch = b）
                for pos in range(p0, p1):
                    if self.perturb_mode == "overwrite":
                        pert = x_chunk.clone()
                        pert[:, pos, :] = self.inject_value
                    else:
                        b = x_chunk.size(0)
                        ext = torch.zeros(b, L + 1, self.C, device=device, dtype=x_chunk.dtype)
                        ext[:, :pos, :] = x_chunk[:, :pos, :]
                        ext[:, pos, :] = self.inject_value
                        ext[:, pos + 1:, :] = x_chunk[:, pos:, :]
                        pert = ext[:, :L, :]

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits = self.model(pert)
                        loss = F.cross_entropy(logits, t_chunk, reduction="mean")

                    eta_sum[pos] += float(loss)

        # 归一化到 [0,1]
        eta = (eta_sum - eta_sum.min()) / (eta_sum.max() - eta_sum.min() + 1e-8)
        self.eta = eta
        # 归一化 tau
        self.tau = (self.tau - self.tau.min()) / (self.tau.max() - self.tau.min() + 1e-8)

    # ---------- 构造解 ----------

    @torch.inference_mode()
    def construct_solution(self) -> List[torch.Tensor]:
        """带权抽样（可重复），权重 ~ tau^alpha * eta^beta。"""
        weights = (self.tau.clamp(min=1e-12) ** self.alpha) * (self.eta.clamp(min=1e-12) ** self.beta)
        probs = (weights / weights.sum()).to(dtype=torch.float32)
        choices = torch.multinomial(probs, num_samples=self.max_inject * self.numant, replacement=True)
        choices = choices.view(self.numant, self.max_inject)
        return [choices[i].clone() for i in range(self.numant)]

    # ---------- 对“小批样本”应用扰动（只对传入块操作） ----------

    @torch.inference_mode()
    def apply_perturbation_chunk(
        self,
        x_chunk: torch.Tensor,          # [b, L, 1]
        solution: torch.Tensor,         # [K]
    ) -> torch.Tensor:
        """仅在给定样本块上施加扰动（避免在全量 B 上预分配大张量）。"""
        sol = torch.as_tensor(solution, device=x_chunk.device, dtype=torch.long)
        if self.perturb_mode == "overwrite":
            out = x_chunk.clone()
            out[:, sol, :] = self.inject_value
            return out
        else:
            # insert_truncate：一次性插入 K 个位置
            b, L, C = x_chunk.shape
            K = sol.numel()
            ext_len = L + K
            mask = torch.zeros(ext_len, device=x_chunk.device, dtype=torch.bool)
            sol_sorted, _ = torch.sort(sol)
            adjusted = sol_sorted + torch.arange(K, device=x_chunk.device)
            mask[adjusted] = True

            ext = torch.empty(b, ext_len, C, device=x_chunk.device, dtype=x_chunk.dtype)
            ext[:, mask, :] = self.inject_value
            ext[:, ~mask, :] = x_chunk
            return ext[:, :L, :]

    # ---------- 评估（分块前向，不构造全量 out） ----------

    @torch.inference_mode()
    def evaluate_solution(
        self,
        solution: torch.Tensor,
        labels: torch.Tensor,
        eval_chunk: int = 1024,
        model = None
    ):
        """对给定 solution 计算全量样本上的平均 loss 与正确数（分块小步前向）。"""
        labels = torch.as_tensor(labels, device=self.device)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        if model is None:
            model = self.model

        for s in range(0, self.B, eval_chunk):
            e = min(s + eval_chunk, self.B)
            x_chunk = self.original_trace[s:e]
            y_chunk = labels[s:e]
            t_chunk = self._to_class_indices(y_chunk)

            pert = self.apply_perturbation_chunk(x_chunk, solution)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(pert)
                loss = F.cross_entropy(logits, t_chunk, reduction="mean")
            total_loss += float(loss) * (e - s)
            total_correct += int((logits.argmax(dim=-1) == t_chunk).sum().item())
            total_seen += (e - s)

        avg_loss = total_loss / max(1, total_seen)
        return avg_loss, total_correct

    # ---------- 主流程 ----------

    @torch.inference_mode()
    def run(
        self,
        labels,
        sample_for_sensitivity: Optional[int] = 1000,
        sample_chunk: int = 512,
        pos_chunk: int = 64,
        eval_chunk: int = 1024,
        verbose: bool = True,
    ):
        """
        - sample_for_sensitivity: 敏感性分析用样本数（强烈建议抽样）
        - 其余为分块超参，可根据显存/速度微调
        """
        self.labels = torch.as_tensor(labels, device=self.device)

        # 1) 抽样做敏感性分析（不要全量 240000）
        if sample_for_sensitivity is None:
            sample_idx = torch.arange(self.B, device=self.device)
        else:
            k = min(int(sample_for_sensitivity), self.B)
            sample_idx = torch.randperm(self.B, device=self.device)[:k]

        self.sensitive_results(
            sample_indices=sample_idx,
            labels=self.labels,
            sample_chunk=sample_chunk,
            pos_chunk=pos_chunk,
        )

        best_fitness = -float("inf")
        best_correct = -float("inf")
        best_solution = None

        # 2) 蚁群主循环（评估均为分块小步前向）
        for it in range(self.max_iter):
            solutions = self.construct_solution()
            fit_list = []
            for sol in solutions:
                avg_loss, correct = self.evaluate_solution(sol, self.labels, eval_chunk=eval_chunk)
                fit_list.append(avg_loss)
                if avg_loss > best_fitness:
                    best_fitness = avg_loss
                    best_correct = correct
                    best_solution = sol.clone()

            # 信息素更新（数值安全 + 归一化）
            fit_t = torch.tensor(fit_list, device=self.device, dtype=self.tau.dtype)
            max_fit = fit_t.max()
            min_fit = fit_t.min()
            self.tau.mul_(1.0 - self.rho)
            denom = (max_fit - min_fit + 1e-8)
            for sol, fit in zip(solutions, fit_t):
                self.tau[sol] += float((fit - min_fit) / denom)
            self.tau = (self.tau - self.tau.min()) / (self.tau.max() - self.tau.min() + 1e-8)

            if verbose:
                atk_success = 1.0 - (best_correct / self.B)
                print(f"[Iter {it}] best_fitness={best_fitness:.6f}  攻击成功样本率={atk_success:.4f}  "
                      f"{self.B - best_correct}/{self.B}")

        self.best_solution = best_solution
        return best_solution, best_fitness
