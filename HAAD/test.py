# main_cross_model_attack.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from Ant_algorithm import UniversalHAAD
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DLWF_pytorch.model.model_5000 import *
from utils.data import load_rimmer_dataset


class CrossModelAttackEvaluator:
    """跨模型攻击评估器"""
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
    def evaluate_model_performance(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        labels: torch.Tensor,
        batch_size: int = 512,
        model_name: str = "Model"
    ) -> Dict:
        """评估模型在数据上的性能"""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size].to(self.device)
                batch_labels = labels[i:i+batch_size].to(self.device)
                
                if batch_labels.ndim > 1 and batch_labels.size(-1) > 1:
                    targets = batch_labels.argmax(dim=-1)
                else:
                    targets = batch_labels.long()
                
                outputs = model(batch_data)
                loss = F.cross_entropy(outputs, targets, reduction='sum')
                
                probs = F.softmax(outputs, dim=-1)
                pred_classes = outputs.argmax(dim=-1)
                
                # 统计
                total_loss += loss.item()
                total_correct += (pred_classes == targets).sum().item()
                total_samples += batch_data.size(0)
                
                # 收集预测和置信度
                predictions.extend(pred_classes.cpu().numpy())
                target_confidences = probs[torch.arange(len(targets)), targets]
                confidences.extend(target_confidences.cpu().numpy())
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        avg_confidence = np.mean(confidences)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'avg_confidence': avg_confidence,
            'total_samples': total_samples
            # 'predictions': np.array(predictions),
            # 'confidences': np.array(confidences)
        }
    
    def compare_performance(
        self, 
        original_results: Dict, 
        perturbed_results: Dict
    ) -> Dict:
        """比较原始和扰动后的性能"""
        return {
            'accuracy_drop': original_results['accuracy'] - perturbed_results['accuracy'],
            'loss_increase': perturbed_results['avg_loss'] - original_results['avg_loss'],
            'confidence_drop': original_results['avg_confidence'] - perturbed_results['avg_confidence'],
            'attack_success_rate': 1.0 - perturbed_results['accuracy'],
            'relative_accuracy_drop': (original_results['accuracy'] - perturbed_results['accuracy']) / original_results['accuracy'] if original_results['accuracy'] > 0 else 0
        }


def load_models(model_paths: Dict[str, str], device: torch.device) -> Dict[str, nn.Module]:
    """加载模型字典"""
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            model = torch.load(path, map_location=device)
            model.to(device)
            model.eval()
            models[name] = model
            print(f"✅ 加载模型 {name}: {path}")
        else:
            print(f"❌ 模型文件不存在: {path}")
    return models


def save_results(results: Dict, save_path: str):
    """保存结果到JSON文件"""
    # 转换numpy数组为列表以便JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    
    json_results = convert_for_json(results)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    
    print(f"✅ 结果已保存到: {save_path}")


def plot_results(results: Dict, save_dir: str):
    """绘制结果可视化图表"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 提取数据
    model_names = list(results['target_models'].keys())
    accuracy_drops = [results['target_models'][name]['comparison']['accuracy_drop'] 
                     for name in model_names]
    loss_increases = [results['target_models'][name]['comparison']['loss_increase'] 
                     for name in model_names]
    confidence_drops = [results['target_models'][name]['comparison']['confidence_drop'] 
                       for name in model_names]
    attack_success_rates = [results['target_models'][name]['comparison']['attack_success_rate'] 
                           for name in model_names]
    
    # 1. 准确率下降
    axes[0, 0].bar(model_names, accuracy_drops, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Drop by Model', fontsize=12, weight='bold')
    axes[0, 0].set_ylabel('Accuracy Drop')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracy_drops):
        axes[0, 0].text(i, v + max(accuracy_drops)*0.01, f'{v:.3f}', ha='center')
    
    # 2. 损失增加
    axes[0, 1].bar(model_names, loss_increases, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Loss Increase by Model', fontsize=12, weight='bold')
    axes[0, 1].set_ylabel('Loss Increase')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(loss_increases):
        axes[0, 1].text(i, v + max(loss_increases)*0.01, f'{v:.3f}', ha='center')
    
    # 3. 置信度下降
    axes[1, 0].bar(model_names, confidence_drops, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Confidence Drop by Model', fontsize=12, weight='bold')
    axes[1, 0].set_ylabel('Confidence Drop')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(confidence_drops):
        axes[1, 0].text(i, v + max(confidence_drops)*0.01, f'{v:.3f}', ha='center')
    
    # 4. 攻击成功率
    axes[1, 1].bar(model_names, attack_success_rates, color='gold', alpha=0.7)
    axes[1, 1].set_title('Attack Success Rate by Model', fontsize=12, weight='bold')
    axes[1, 1].set_ylabel('Attack Success Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(attack_success_rates):
        axes[1, 1].text(i, v + max(attack_success_rates)*0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, 'attack_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化图表已保存到: {plot_path}")
    
    # 绘制适应度历史
    if 'perturbation_info' in results and 'fitness_history' in results['perturbation_info']:
        plt.figure(figsize=(10, 6))
        fitness_history = results['perturbation_info']['fitness_history']
        plt.plot(fitness_history, 'b-', linewidth=2, alpha=0.8)
        plt.title('Fitness Evolution During ACO Optimization', fontsize=14, weight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.grid(True, alpha=0.3)
        
        fitness_plot_path = os.path.join(save_dir, 'fitness_evolution.png')
        plt.savefig(fitness_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 适应度进化图已保存到: {fitness_plot_path}")


def main():
    """主函数：跨模型通用扰动攻击实验"""
    
    # ==================== 配置参数 ====================
    config = {
        # 数据配置
        'data_path': '/home/xuke/lpf/HAAD-GA/utils/Dataset/Rimmer/tor_100w_2500tr.npz',  # 你的数据文件路径
        'batch_size': 512,
        
        # 模型配置 - 替换为你的实际模型路径
        'source_model_path': '/home/xuke/lpf/HAAD-GA/DLWF_pytorch/trained_model/length_5000/Rimmer/llm_rimmer100.pth',  # 源模型llm（用于生成扰动）
        # 'source_model_path':'/home/xuke/lpf/HAAD-GA/DLWF_pytorch/trained_model/length_5000/Rimmer/df_rimmer100.pth',
        'target_models': {
            'df': '/home/xuke/lpf/HAAD-GA/DLWF_pytorch/trained_model/length_5000/Rimmer/df_rimmer100.pth',        # 目标模型B
            'varcnn': '/home/xuke/lpf/HAAD-GA/DLWF_pytorch/trained_model/length_5000/Rimmer/varcnn_rimmer100.pth',        # 目标模型C（可选）
            # 'llm':'/home/xuke/lpf/HAAD-GA/DLWF_pytorch/trained_model/length_5000/Rimmer/llm_rimmer100.pth',
        },
        
        # HAAD算法配置
        'haad_config': {
            'numant': 20,                    # 蚂蚁数量
            'max_inject': 200,                # 最大扰动位置数
            'max_iter': 10,                 # 最大迭代数
            'alpha': 1.0,                    # 信息素重要性
            'beta': 2.0,                     # 启发式信息重要性
            'rho': 0.1,                      # 信息素蒸发率
            'inject_value': 1.0,             # 扰动值
            'perturb_mode': 'overwrite',     # 扰动模式
            'memory_limit_mb': 8192,         # 显存限制
            'eval_chunk_size': 256,          # 评估批量大小
        },
        
        # 实验配置
        'fitness_type': 'accuracy_drop',     # 适应度函数类型
        'heuristic_sample_size': 2000,       # 启发式信息采样大小
        'pheromone_strategy': 'elitist',     # 信息素更新策略
        
        # 输出配置
        'output_dir': 'results/cross_model_attack',
        'save_plots': True,
        'verbose': True
    }
    
    # ==================== 初始化 ====================
    print("🚀 开始跨模型通用扰动攻击实验")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config['output_dir']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = CrossModelAttackEvaluator(device)
    
    # ==================== 加载数据 ====================
    print("\n📁 加载数据...")
    
    # 这里需要根据你的数据格式进行调整
    if os.path.exists(config['data_path']):
        # data_dict = torch.load(config['data_path'])
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_rimmer_dataset(input_size=5000,num_classes=100,test_ratio=0.1,val_ratio=0.1)
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.int)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.int)
        
        print(f"✅ 数据加载成功")
        print(f"   训练数据: {train_data.shape}")
        print(f"   测试数据: {test_data.shape}")
    else:
        # 示例：生成模拟数据
        print("⚠️  数据文件不存在，生成模拟数据")
        B, L, C = 1000, 100, 1
        train_data = torch.randn(B, L, C)
        train_labels = torch.randint(0, 10, (B,))
        test_data = torch.randn(200, L, C)
        test_labels = torch.randint(0, 10, (200,))
    
    # ==================== 加载模型 ====================
    print("\n🤖 加载模型...")
    
    # 加载源模型A（用于生成扰动）
    if os.path.exists(config['source_model_path']):
        source_model =  MultiScaleLLM_V2(num_classes=100).to(device)  # 替换为你的模型类
        # source_model = DFNet(100).to(device)
        source_model.load_state_dict(torch.load(config['source_model_path'], map_location=device))
        source_model.to(device)
        source_model.eval()
        print(f"✅ 源模型A加载成功: {config['source_model_path']}")
    else:
        # 示例：创建模拟模型
        print("⚠️  源模型文件不存在，创建模拟模型")
        source_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(train_data.shape[1] * train_data.shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)
    
    # 加载目标模型B、C、D等
    target_models = {}
    for name, path in config['target_models'].items():
        if os.path.exists(path):
            if name == 'df':
                model = DFNet(100).to(device)
            elif name == 'varcnn':
                model = VarCNN(5000,100).to(device)
            elif name == 'llm':
                model = MultiScaleLLM_V2(num_classes=100).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            target_models[name] = model
            print(f"✅ 目标模型{name}加载成功: {path}")

    
    # ==================== 步骤1: 用源模型A生成通用扰动 ====================
    print("\n🐜 步骤1: 使用源模型A运行蚁群算法生成通用扰动...")
    print("-" * 30)
    
    # 初始化HAAD算法
    haad = UniversalHAAD(
        model=source_model,
        original_trace=train_data,
        **config['haad_config']
    )
    
    # 运行优化
    best_positions, best_fitness = haad.run(
        labels=train_labels,
        fitness_type=config['fitness_type'],
        heuristic_sample_size=config['heuristic_sample_size'],
        pheromone_strategy=config['pheromone_strategy'],
        verbose=config['verbose']
    )
    
    # 获取扰动信息
    perturbation_info = haad.get_universal_perturbation_info()
    
    print(f"\n✅ 通用扰动生成完成!")
    print(f"🎯 扰动位置: {perturbation_info['perturbation_positions']}")
    print(f"📊 扰动位置数量: {perturbation_info['num_positions']}")
    print(f"📈 最终适应度: {perturbation_info['fitness']:.6f}")
    print(f"📉 扰动比例: {perturbation_info['perturbation_ratio']:.2%}")
    
    # ==================== 步骤2: 在源模型A上验证扰动效果 ====================
    print(f"\n🔍 步骤2: 在源模型A上验证扰动效果...")
    print("-" * 30)
    
    # 原始性能
    source_original = evaluator.evaluate_model_performance(
        source_model, test_data, test_labels, 
        config['batch_size'], "Source_Model_A"
    )
    
    # 扰动后性能
    perturbed_test_data = haad.apply_to_new_data(test_data)
    source_perturbed = evaluator.evaluate_model_performance(
        source_model, perturbed_test_data, test_labels, 
        config['batch_size'], "Source_Model_A_Perturbed"
    )
    
    source_comparison = evaluator.compare_performance(source_original, source_perturbed)
    
    print(f"📊 源模型A性能对比:")
    print(f"   原始准确率: {source_original['accuracy']:.4f}")
    print(f"   扰动后准确率: {source_perturbed['accuracy']:.4f}")
    print(f"   准确率下降: {source_comparison['accuracy_drop']:.4f}")
    print(f"   攻击成功率: {source_comparison['attack_success_rate']:.4f}")
    
    # ==================== 步骤3: 在目标模型B等上验证迁移效果 ====================
    print(f"\n🔄 步骤3: 在目标模型上验证迁移攻击效果...")
    print("-" * 30)
    
    target_results = {}
    
    for model_name, model in target_models.items():
        print(f"\n📋 评估目标模型: {model_name}")
        
        # 原始性能
        original_perf = evaluator.evaluate_model_performance(
            model, test_data, test_labels, 
            config['batch_size'], model_name
        )
        
        # 扰动后性能
        perturbed_perf = evaluator.evaluate_model_performance(
            model, perturbed_test_data, test_labels, 
            config['batch_size'], f"{model_name}_Perturbed"
        )
        
        # 性能对比
        comparison = evaluator.compare_performance(original_perf, perturbed_perf)
        
        target_results[model_name] = {
            'original': original_perf,
            'perturbed': perturbed_perf,
            'comparison': comparison
        }
        
        print(f"   原始准确率: {original_perf['accuracy']:.4f}")
        print(f"   扰动后准确率: {perturbed_perf['accuracy']:.4f}")
        print(f"   准确率下降: {comparison['accuracy_drop']:.4f}")
        print(f"   攻击成功率: {comparison['attack_success_rate']:.4f}")
        print(f"   迁移效果: {'✅ 良好' if comparison['accuracy_drop'] > 0.1 else '⚠️ 一般' if comparison['accuracy_drop'] > 0.05 else '❌ 较差'}")
    
    # ==================== 步骤4: 汇总结果 ====================
    print(f"\n📈 步骤4: 汇总实验结果...")
    print("=" * 50)
    
    # 整理所有结果
    final_results = {
        'experiment_info': {
            'timestamp': timestamp,
            'config': config,
            'device': str(device)
        },
        'perturbation_info': perturbation_info,
        'source_model': {
            'original': source_original,
            'perturbed': source_perturbed,
            'comparison': source_comparison
        },
        'target_models': target_results,
        'summary': {
            'avg_accuracy_drop': np.mean([res['comparison']['accuracy_drop'] 
                                        for res in target_results.values()]),
            'avg_attack_success_rate': np.mean([res['comparison']['attack_success_rate'] 
                                              for res in target_results.values()]),
            'best_transfer_model': max(target_results.keys(), 
                                     key=lambda x: target_results[x]['comparison']['accuracy_drop']),
            'worst_transfer_model': min(target_results.keys(), 
                                      key=lambda x: target_results[x]['comparison']['accuracy_drop'])
        }
    }
    
    # 打印总结
    print(f"🎯 实验总结:")
    print(f"   扰动位置数量: {perturbation_info['num_positions']}")
    print(f"   扰动比例: {perturbation_info['perturbation_ratio']:.2%}")
    print(f"   源模型攻击成功率: {source_comparison['attack_success_rate']:.4f}")
    print(f"   平均迁移攻击成功率: {final_results['summary']['avg_attack_success_rate']:.4f}")
    print(f"   最佳迁移模型: {final_results['summary']['best_transfer_model']}")
    print(f"   最差迁移模型: {final_results['summary']['worst_transfer_model']}")
    
    # ==================== 步骤5: 保存结果 ====================
    print(f"\n💾 步骤5: 保存实验结果...")
    
    # 保存JSON结果
    results_path = os.path.join(output_dir, 'experiment_results.json')
    save_results(final_results, results_path)
    
    # 保存扰动位置
    perturbation_path = os.path.join(output_dir, 'universal_perturbation.pt')
    torch.save({
        'positions': best_positions,
        'fitness': best_fitness,
        'config': config['haad_config']
    }, perturbation_path)
    print(f"✅ 通用扰动已保存到: {perturbation_path}")
    
    # 保存配置
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ 实验配置已保存到: {config_path}")
    
    # 生成可视化图表
    if config['save_plots']:
        print(f"\n, 生成可视化图表...")
        plot_results(final_results, output_dir)
    
    print(f"\n🎉 实验完成! 所有结果已保存到: {output_dir}")
    print("=" * 50)
    
    return final_results


if __name__ == "__main__":
    results = main()